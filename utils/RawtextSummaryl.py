import time, re

class SummaryCouncil:
    def __init__(self, client, model_manager):
        self.client = client
        self.mm = model_manager
        self.active_idx = 0  # Ghi nhớ model đang chạy tốt

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
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            return lines[-1] if lines else ""
        except Exception as e:
            if "404" in str(e) or "resources" in str(e).lower():
                # CHỈ KHI LỖI: Đánh dấu lỗi và báo hiệu để hàm chính nhảy sang model tiếp theo
                self.mm.mark_failed(model_name)
                return "FAIL_RESOURCE"
            return None

    def process_summary(self, post_id, content_block):
        # Lấy full_text giống DiseaseCouncil
        input_text = content_block.get('full_text', '')
        if not input_text: return "NA"

        system_prompt = "Summarize the vaccine concern in ONE professional sentence. Output ONLY the sentence."

        res = None
        while res is None or res == "FAIL_RESOURCE":
            model_name = self.mm.get_model("CLIENT", attempt=self.active_idx)
            if not model_name: return "NA"

            res = self._execute_llm_call(model_name, "CLIENT", system_prompt, input_text)

            if res == "FAIL_RESOURCE":
                self.active_idx += 1  # Chuyển sang model kế tiếp vĩnh viễn
            elif res:
                return res
        return "NA"