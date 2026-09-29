import time, re, json


class NarrativeCouncil:
    def __init__(self, client, model_manager, taxonomy_data):
        self.client = client
        self.mm = model_manager
        self.taxonomy = taxonomy_data["uns_taxonomy_config"]
        self.active_idx = 0
        self.knowledge_base = self._prepare_knowledge_base()

    def _prepare_knowledge_base(self):
        """Chuyển taxonomy thành chuỗi văn bản để AI tham chiếu."""
        kb = []
        for sub in self.taxonomy["subtopics"]:
            sub_name = sub["subtopic_name"]
            for nar in sub["narratives"]:
                kb.append(
                    f"- Subtopic: {sub_name} | Narrative: {nar['specific_narrative']} | "
                    f"FLICC: {nar['flicc']} | Stigma: {nar['stigma_target']} | Trigger: {nar['real_world_trigger']}"
                )
        return "\n".join(kb)

    def _execute_llm_call(self, model_name, role, system_prompt, user_prompt):
        params = self.mm.get_params(role)
        timeout = self.mm.get_timeout(model_name)
        time.sleep(2.0)
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                **params, timeout=timeout
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if "404" in str(e) or "resources" in str(e).lower():
                self.mm.mark_failed(model_name)
                return "FAIL_RESOURCE"
            return "NA"

    def _screen_for_misinformation(self, summary_text):
        """
        Pre-screening: Kiểm tra xem post có chứa misinformation không
        trước khi gán taxonomy. Trả về True nếu có misinformation.
        """
        # Rule-based pre-check: detect known misinformation patterns
        # BEFORE calling LLM — catches cases where phrasing is indirect
        summary_lower = summary_text.lower()
        MISINFO_SIGNALS = [
            # Vaccine-autism claims
            "vaccines cause autism", "vaccine injury and autism", "vaccine-autism",
            "autism diagnos", "wakefield",
            # Toxic/dangerous ingredient claims
            "toxic spike protein", "spike protein", "aluminum adjuvant",
            "aluminum in vaccine", "adjuvant", "graphene",
            # Safety/study concerns framed as misinformation
            "lack of long-term safety", "no long-term studies",
            "safety studies on the", "not been adequately tested",
            "long-term safety studies", "comprehensive safety",
            "long-term safety", "comprehensive, long-term",
            # Suppression/conspiracy
            "authors have not disclosed", "hiding data", "retracted",
            "calls for retraction", "suppressed", "cover-up",
            "study was retracted",
            # Exaggerated harm
            "autoimmune disorder", "neurological damage",
            "toxic spike", "inflammatory cytokine",
            "persistent elevation", "ongoing immune stimulation",
            "adverse reaction", "death from vaccine",
            # mRNA conspiracy
            "toxic mrna", "mrna injection", "gene therapy",
            "alters dna", "alter your dna", "changes dna",
            # Trial exclusion claims
            "excluded individuals", "excluded from trials",
            "clinical trial excluded",
        ]
        for signal in MISINFO_SIGNALS:
            if signal in summary_lower:
                # Guard: if post is DEBUNKING the misinformation, return False
                # e.g. "retracted and discredited" = debunking Wakefield
                is_debunking = any(phrase in summary_lower for phrase in [
                    "retracted and discredited",
                    "debunked",
                    "falsified data",
                    "no credible evidence",
                    "scientific consensus",
                    "disproven",
                ])
                if is_debunking:
                    return False
                return True

        screen_prompt = (
            "You are a fact-checker specializing in vaccine misinformation.\n"
            "Read the following vaccine-related summary.\n"
            "Answer ONLY 'YES' or 'NO':\n\n"
            "Answer 'YES' if the summary contains ANY of:\n"
            "- False or misleading claims about vaccine safety or ingredients\n"
            "- Claims linking vaccines to autism, cancer, infertility, or death\n"
            "- Conspiracy theories about vaccine manufacturers or health authorities\n"
            "- Pseudoscientific claims (toxic spike proteins, graphene, DNA alteration)\n"
            "- Exaggerated harm claims not supported by scientific consensus\n"
            "- Suppression of data or research manipulation claims\n"
            "- Anti-vaccine rhetoric disguised as safety concern\n"
            "- A study cited to support anti-vaccine claims\n\n"
            "Answer 'NO' ONLY if the post is:\n"
            "- A neutral question about vaccine schedules, dosing, or timing\n"
            "- A personal report of KNOWN/EXPECTED vaccine side effects (fever, soreness)\n"
            "- A request for medical advice or travel vaccine information\n"
            "- Pro-vaccine or vaccine access advocacy content\n"
            "- Scientific correction of misinformation (debunking)\n"
            "- A question about vaccine record-keeping or insurance coverage\n\n"
            f"Summary: {summary_text}\n\n"
            "Answer (YES or NO only):"
        )

        model_name = self.mm.get_model("CLIENT", attempt=self.active_idx)
        if not model_name:
            return True  # Default proceed jika model tidak tersedia

        res = self._execute_llm_call(model_name, "CLIENT", "", screen_prompt)
        if res and "YES" in res.upper():
            return True
        return False

    def _rerun_with_stricter_prompt(self, summary_text, model_name):
        """
        Re-run khi detect được false positive combo:
        False Dichotomy + Zero Protection Claims + Public Health Officials
        """
        strict_prompt = (
            "IMPORTANT: Re-evaluate this post carefully.\n\n"
            "The combination 'False Dichotomy + Zero Protection Claims + Public Health Officials' "
            "should ONLY be assigned when the post EXPLICITLY argues that vaccines provide "
            "ZERO protection OR directly attacks public health authorities as corrupt/dishonest.\n\n"
            "If this post is a neutral vaccine question, schedule inquiry, factual medical discussion, "
            "or personal health concern — output the neutral JSON below.\n\n"
            "NEUTRAL OUTPUT (use this if no clear misinformation):\n"
            '{"Subtopic": "None", "Narrative": "None", "FLICC": "None", '
            '"Stigma": "None", "Trigger": "None"}\n\n'
            f"Summary: {summary_text}\n\n"
            "Output JSON only, no explanation:"
        )
        res = self._execute_llm_call(model_name, "CLIENT", "", strict_prompt)
        try:
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return None

    def process_narrative(self, post_id, summary_text):
        """Phân tích Summary và trả về dictionary chứa toàn bộ các cột phân loại."""
        if not summary_text or summary_text == "NA":
            return self._empty_row()

        # STEP 1: Pre-screening — block neutral posts trước khi classify
        is_misinfo = self._screen_for_misinformation(summary_text)
        if not is_misinfo:
            return self._neutral_row()

        # STEP 2: Classify với prompt chặt hơn
        shared_instruction = (
            "You are a Public Health Expert specializing in vaccine misinformation detection.\n"
            "Analyze the Summary and match it to the Taxonomy ONLY if it contains clear "
            "misinformation, conspiracy theories, or harmful anti-vaccine narratives.\n\n"
            f"TAXONOMY REFERENCE:\n{self.knowledge_base}\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY a valid JSON with keys: Subtopic, Narrative, FLICC, Stigma, Trigger.\n"
            "2. Values MUST match the Taxonomy Reference EXACTLY — do not invent new values.\n"
            "3. If the post is a neutral question, personal symptom report, factual vaccine info, "
            "or legitimate health concern WITHOUT misinformation, output:\n"
            '   {"Subtopic": "None", "Narrative": "None", "FLICC": "None", '
            '"Stigma": "None", "Trigger": "None"}\n'
            "4. DO NOT assign 'False Dichotomy', 'Zero Protection Claims', or "
            "'Public Health Officials' to posts that merely ask vaccine schedule questions, "
            "report personal side effects, or seek medical advice.\n"
            "5. 'Turbo Cancer' narrative ONLY applies when the post EXPLICITLY claims "
            "vaccines cause cancer — not for general health questions.\n"
            "6. 'Contextomy' ONLY applies when someone is deliberately misquoting or "
            "removing context from a study/statement — not for personal symptom reports.\n"
            "7. 'Celebrity/Athlete Collapse' trigger ONLY applies when the post "
            "references a specific public figure collapsing — not for general vaccine questions.\n"
            "8. 'Ingredient Safety Concerns' narrative applies when the post claims vaccine "
            "ingredients (aluminum adjuvants, spike proteins, mRNA, graphene) cause long-term harm, "
            "autoimmune disorders, neurological damage, or inflammatory responses — "
            "even if framed as a 'recent study' or 'research finding'.\n"
            "9. 'Suppression of Data' narrative applies when the post claims researchers, "
            "pharmaceutical companies, or authorities are hiding, manipulating, or failing to "
            "disclose vaccine safety data — including calls for retraction, undisclosed corrections, "
            "or exclusion of vulnerable groups from clinical trials.\n"
            "10. 'Vaccine Safety and Efficacy' narrative applies when the post broadly questions "
            "vaccine safety due to lack of long-term studies, premature approval, or inadequate "
            "clinical trial design — framed as a systemic concern rather than a personal experience."
        )

        res = "FAIL_RESOURCE"
        while res == "FAIL_RESOURCE":
            model_name = self.mm.get_model("CLIENT", attempt=self.active_idx)
            if not model_name:
                break

            res = self._execute_llm_call(
                model_name, "CLIENT", shared_instruction,
                f"Summary: {summary_text}"
            )

            if res == "FAIL_RESOURCE":
                self.active_idx += 1
            else:
                try:
                    match = re.search(r'\{.*\}', res, re.DOTALL)
                    if match:
                        # FIX: json.loads thay vì eval() — an toàn hơn
                        data = json.loads(match.group(0))

                        # STEP 3: Nếu model trả về None cho tất cả → neutral row
                        none_values = ("None", "none", "", "NA", None)
                        if all(data.get(k) in none_values
                               for k in ["Subtopic", "Narrative", "FLICC", "Stigma"]):
                            return self._neutral_row()

                        # STEP 4: Block known false positive combo
                        flicc = str(data.get("FLICC", "None"))
                        narrative = str(data.get("Narrative", "None"))
                        stigma = str(data.get("Stigma", "None"))

                        is_fp_combo = (
                            "False Dichotomy" in flicc
                            and "Zero Protection" in narrative
                            and "Public Health" in stigma
                        )
                        if is_fp_combo:
                            rerun_data = self._rerun_with_stricter_prompt(
                                summary_text, model_name
                            )
                            if rerun_data is None:
                                return self._neutral_row()
                            data = rerun_data

                            # Nếu sau re-run vẫn là None → neutral
                            if all(data.get(k) in none_values
                                   for k in ["Subtopic", "Narrative", "FLICC", "Stigma"]):
                                return self._neutral_row()

                        # STEP 5: Post-classification validation
                        # Catch spurious narrative/trigger assignments
                        narrative_out = str(data.get("Narrative", "None"))
                        trigger_out   = str(data.get("Trigger", "None"))
                        summary_lower = summary_text.lower()

                        # Guard: Turbo Cancer only if cancer explicitly mentioned
                        if any(t in narrative_out for t in
                               ["Turbo Cancer", "Chronic Disease (Turbo Cancer)"]):
                            if not any(kw in summary_lower for kw in
                                       ["cancer", "tumor", "tumour", "turbo"]):
                                return self._neutral_row()

                        # Guard: Alternative Cures only if alternative treatment mentioned
                        if "Alternative Cures" in narrative_out:
                            if not any(kw in summary_lower for kw in [
                                "alternative", "natural", "herbal", "remedy",
                                "homeopathy", "instead of vaccine", "cure",
                                "ivermectin", "hydroxychloroquine",
                            ]):
                                return self._neutral_row()

                        # Guard: Celebrity/Athlete Collapse only if public figure mentioned
                        if "Celebrity" in trigger_out or "Athlete" in trigger_out:
                            if not any(kw in summary_lower for kw in [
                                "celebrity", "athlete", "famous", "public figure",
                                "collapsed", "died suddenly", "sudden death",
                            ]):
                                data["Trigger"] = "None"

                        return {
                            "Category ID":              self.taxonomy["category_id"],
                            "Category Name":            self.taxonomy["category_name"],
                            "Topic":                    self.taxonomy["topic"],
                            "Subtopic":                 data.get("Subtopic", "None"),
                            "Specific Narrative":       data.get("Narrative", "None"),
                            "Cognitive Tactic (FLICC)": data.get("FLICC", "None"),
                            "Stigma Target":            data.get("Stigma", "None"),
                            "Real-World Trigger":       data.get("Trigger", "None")
                        }

                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

        return self._empty_row()

    def _neutral_row(self):
        """Row cho posts không có misinformation."""
        return {
            "Category ID":              self.taxonomy["category_id"],
            "Category Name":            self.taxonomy["category_name"],
            "Topic":                    self.taxonomy["topic"],
            "Subtopic":                 "None",
            "Specific Narrative":       "None",
            "Cognitive Tactic (FLICC)": "None",
            "Stigma Target":            "None",
            "Real-World Trigger":       "None"
        }

    def _empty_row(self):
        return {k: "NA" for k in [
            "Category ID", "Category Name", "Topic", "Subtopic",
            "Specific Narrative", "Cognitive Tactic (FLICC)",
            "Stigma Target", "Real-World Trigger"
        ]}