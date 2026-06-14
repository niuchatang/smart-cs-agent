from intent.message_signals import is_casual_chitchat, is_travel_knowledge_query
from main import CustomerServiceAgent


def test_casual_chitchat_detection():
    assert is_casual_chitchat("你好呀，今天心情不太好，随便聊几句吧")
    assert not is_travel_knowledge_query("你好呀，今天心情不太好，随便聊几句吧")


def test_render_reply_skips_irrelevant_rag_when_llm_disabled():
    agent = CustomerServiceAgent()
    agent.llm = None
    agent.answer_chain = None
    msg = "你好呀，今天心情不太好，随便聊几句吧"
    rag = agent.rag_store.retrieve(msg, k=3)
    reply = agent._render_reply("unknown", msg, [], "", rag)
    assert "机场快线" not in reply
    assert "智慧交通客服" in reply


def test_render_reply_uses_llm_reply_for_casual_chat():
    agent = CustomerServiceAgent()
    msg = "你好呀，今天心情不太好，随便聊几句吧"
    reply = agent._render_reply("unknown", msg, [], "你好呀，愿意陪你聊聊。", [])
    assert reply == "你好呀，愿意陪你聊聊。"
