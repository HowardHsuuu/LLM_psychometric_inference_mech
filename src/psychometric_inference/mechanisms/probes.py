"""
40 general probe questions for contrastive activation extraction.

Requirements:
- Unrelated to any specific psychological scale (no direct overlap with IRI, PANAS, etc.)
- Open-ended enough to elicit personality-revealing responses
- In Traditional Chinese (matching the Taiwanese college student persona)
- Cover diverse life domains to sample broad personality expression

These are NOT psychological assessments — they're neutral prompts designed to
elicit natural-language responses whose activations will differ based on the
persona condition in the system prompt.
"""

PROBE_QUESTIONS = [
    # 日常生活
    "描述你典型的一個週末是怎麼度過的。",
    "你最近讀了什麼書或看了什麼電影？說說你的感想。",
    "你通常怎麼決定晚餐要吃什麼？",
    "描述你的房間或書桌的樣子。",
    "你有什麼固定的日常習慣嗎？",

    # 社交與人際
    "你跟朋友意見不合的時候，通常會怎麼處理？",
    "描述你最好的朋友是什麼樣的人。",
    "你在社交場合中通常扮演什麼角色？",
    "你覺得維持友誼最重要的是什麼？",
    "如果朋友突然取消了跟你的約定，你會怎麼反應？",

    # 自我反思
    "你覺得自己最大的優點是什麼？",
    "有什麼事情是你一直想改變但還沒做到的？",
    "你通常怎麼做重要的決定？",
    "描述一個讓你感到驕傲的經歷。",
    "你覺得別人對你的第一印象通常是什麼？",

    # 壓力與情緒
    "你壓力大的時候通常會做什麼來放鬆？",
    "描述一個最近讓你感到困擾的事情。",
    "你怎麼看待失敗或挫折？",
    "當你心情不好的時候，你會跟別人說嗎？",
    "你覺得什麼事情最容易讓你感到煩躁？",

    # 未來與目標
    "你對五年後的自己有什麼期望？",
    "你選擇現在這個科系的原因是什麼？",
    "你覺得什麼樣的工作最適合你？",
    "如果不用考慮現實，你最想做什麼？",
    "你對「成功」的定義是什麼？",

    # 價值觀
    "你覺得什麼對你來說最重要？",
    "你怎麼看待競爭？",
    "你覺得人與人之間最重要的是什麼？",
    "如果你可以改變社會的一件事，你會改變什麼？",
    "你怎麼看待冒險和安穩之間的平衡？",

    # 具體情境
    "如果你在路上看到有人需要幫助，你會怎麼做？",
    "你參加一個都是陌生人的聚會，你會怎麼表現？",
    "如果有人批評你的作品或想法，你會怎麼回應？",
    "你怎麼安排考試前的準備時間？",
    "如果室友的生活習慣跟你很不同，你會怎麼處理？",

    # 興趣與偏好
    "你最喜歡什麼類型的音樂或藝術？為什麼？",
    "你比較喜歡獨處還是跟朋友在一起？為什麼？",
    "描述一個你覺得最理想的旅行方式。",
    "你有什麼特別的興趣或嗜好嗎？",
    "你比較喜歡計劃好所有事情，還是隨興行事？",
]

assert len(PROBE_QUESTIONS) == 40, f"Expected 40 questions, got {len(PROBE_QUESTIONS)}"