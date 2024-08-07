single_criteria = "Your assessment should focus primarily on the consistency between the assistant's answer and the reference answer."

multi_turn_criteria = "Your assessment should focus on the overall quality of the responses based on the following criteria:\n\
Accuracy: Evaluate the correctness and reliability of the information provided.\
Coherence: Assess the clarity and logical flow of the responses.\
Relevance: Determine how closely each response addresses the question asked.\
Thoroughness: Judge the depth and completeness of the response in covering the topic."

prompts = {
    "single": {
        "system_prompt": f"Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. \nRequirements: {single_criteria} \nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[Rating]]\", for example: \"Rating: [[5]]\".\n\n",
        "prompt_template": "[Question]\n{input}\n\n[The Start of Reference Answer]\n{reference}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{output}\n[The End of Assistant's Answer]",
        "output_format": "[[rating]]"
    },
    "multi_turn_1":{
        "system_prompt": f"Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. \nRequirements: {multi_turn_criteria} \nYou will be given the assistant's answer and some reference. The reference consists of QA pairs related to the patient, which are completely accurate and can be used as a reliable source of truth.\nYou evaluation should focus on the assistant's answer to the first question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[Rating]]\", for example: \"Rating: [[5]]\".\n\n",
        "prompt_template": "<|The Start of Reference|>\n\n{reference}\n\n<|The End of Reference|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{turn_1_question}\n\n### Assistant A:\n{turn_1_answer}\n\n<|The End of Assistant A's Conversation with User|>",
        "output_format": "[[Rating]]"
    },
    "multi_turn_2":{
        "system_prompt": f"Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. \nRequirements: {multi_turn_criteria} \nYou will be given the assistant's answer and some reference. The reference consists of QA pairs related to the patient, which are completely accurate and can be used as a reliable source of truth.\nYou evaluation should focus on the assistant's answer to the second question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[Rating]]\", for example: \"Rating: [[5]]\".\n\n",
        "prompt_template": "<|The Start of Reference|>\n\n{reference}\n\n<|The End of Reference|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{turn_1_question}\n\n### Assistant A:\n{turn_1_answer}\n\n### User:\n{turn_2_question}\n\n### Assistant A:\n{turn_2_answer}\n\n<|The End of Assistant A's Conversation with User|>",
        "output_format": "[[Rating]]"
    },
    "multi_turn_3":{
        "system_prompt": f"Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. \nRequirements: {multi_turn_criteria} \nYou will be given the assistant's answer and some reference. The reference consists of QA pairs related to the patient, which are completely accurate and can be used as a reliable source of truth.\nYou evaluation should focus on the assistant's answer to the third question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[Rating]]\", for example: \"Rating: [[5]]\".\n\n",
        "prompt_template": "<|The Start of Reference|>\n\n{reference}\n\n<|The End of Reference|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{turn_1_question}\n\n### Assistant A:\n{turn_1_answer}\n\n### User:\n{turn_2_question}\n\n### Assistant A:\n{turn_2_answer}\n\n### User:\n{turn_3_question}\n\n### Assistant A:\n{turn_3_answer}\n\n<|The End of Assistant A's Conversation with User|>",
        "output_format": "[[Rating]]"
    }
}




