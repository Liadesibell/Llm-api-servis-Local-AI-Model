from ink_ai.conversations import ConversationStore


def test_conversations_and_file_context_survive_store_restart(tmp_path):
    database_path = tmp_path / "data" / "conversations.sqlite3"
    store = ConversationStore(database_path)
    first_id = store.create_conversation("Kod incelemesi")
    store.add_message(
        first_id,
        "user",
        "Bu dosyayı incele.\n\n[Ek dosya: app.py]",
    )
    store.set_files(
        first_id,
        [{"name": "app.py", "content": "print('hello')"}],
    )
    store.set_images(
        first_id,
        [{"name": "clipboard.png", "content_base64": "iVBORw0KGgo="}],
    )
    store.add_message(first_id, "assistant", "Dosya kısa bir Python betiği.")
    second_id = store.create_conversation("Yeni sohbet")
    store.add_message(second_id, "user", "Merhaba")

    reopened_store = ConversationStore(database_path)

    assert reopened_store.list_conversations()[0] == (second_id, "Yeni sohbet")
    assert reopened_store.get_messages(first_id) == [
        ("user", "Bu dosyayı incele.\n\n[Ek dosya: app.py]"),
        ("assistant", "Dosya kısa bir Python betiği."),
    ]
    assert reopened_store.get_files(first_id) == [
        {"name": "app.py", "content": "print('hello')"}
    ]
    assert reopened_store.get_images(first_id) == [
        {"name": "clipboard.png", "content_base64": "iVBORw0KGgo="}
    ]
    assert reopened_store.get_messages(second_id) == [("user", "Merhaba")]


def test_search_finds_chat_by_title_or_message(tmp_path):
    store = ConversationStore(tmp_path / "search.sqlite3")
    conversation_id = store.create_conversation("Yeni sohbet")
    store.add_message(conversation_id, "user", "Bir daire çizmek için Python kodu")

    assert store.search_conversations("daire") == [(conversation_id, "Yeni sohbet")]
    assert store.search_conversations("Python") == [(conversation_id, "Yeni sohbet")]
    assert store.search_conversations("olmayan") == []
