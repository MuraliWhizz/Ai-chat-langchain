import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda

import query
from ingest import chunk_text
from rag_utils import EMBEDDING_DIMENSIONS, embed_texts, vector_literal


class ChunkTextTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self) -> None:
        self.assertEqual(chunk_text("  alpha\nbeta  "), ["alpha\nbeta"])

    def test_long_text_obeys_size_and_overlaps(self) -> None:
        text = "a" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=100)

        self.assertEqual([len(chunk) for chunk in chunks], [1000, 1000, 700])
        self.assertEqual(chunks[0][-100:], chunks[1][:100])
        self.assertEqual(chunks[1][-100:], chunks[2][:100])

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)


class VectorLiteralTests(unittest.TestCase):
    def test_serializes_expected_dimension(self) -> None:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        self.assertTrue(vector_literal(vector).startswith("[0.0,"))
        self.assertTrue(vector_literal(vector).endswith("]"))

    def test_rejects_wrong_dimension(self) -> None:
        with self.assertRaises(ValueError):
            vector_literal([1.0])


class LangChainIntegrationTests(unittest.TestCase):
    def test_history_keeps_follow_up_and_prior_source_context(self) -> None:
        history = [
            {"role": "user", "content": "Explain about Muralidhar Badisa"},
            {
                "role": "assistant",
                "content": "A previous answer with irrelevant narration.",
                "sources": [
                    {"source": "/docs/other.pdf", "chunk": 1},
                    {"source": "/docs/MURALIDHAR_BADISA_FA.pdf", "chunk": 4},
                ],
            }
        ]

        formatted = query.format_history(history)

        self.assertIn("Muralidhar Badisa", formatted)
        self.assertNotIn("irrelevant narration", formatted)
        self.assertEqual(
            query.referenced_source("According to source 2, explain him", history),
            "/docs/MURALIDHAR_BADISA_FA.pdf",
        )
        self.assertIn(
            "Muralidhar Badisa",
            query.retrieval_terms("What did he study?", history),
        )

    def test_explicit_follow_up_name_drops_stale_comparison_question(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Compare Muralidhar and Jagadeesh for full-stack roles.",
            },
            {
                "role": "assistant",
                "content": "Muralidhar is the stronger fit.",
                "sources": [
                    {
                        "source": "/docs/MURALIDHAR_BADISA_FA.pdf",
                        "chunk": 0,
                        "similarity": 0.8,
                    },
                    {
                        "source": "/docs/JAGADEESH_CHILAKABATHULA_C.pdf",
                        "chunk": 0,
                        "similarity": 0.7,
                    },
                ],
            },
        ]

        self.assertEqual(
            query.format_history(history, "give some more brief about muralidhar"),
            "",
        )
        self.assertEqual(
            query.format_history(history, "tell me more about him"),
            "user: Compare Muralidhar and Jagadeesh for full-stack roles.",
        )
        self.assertIn(
            "do not repeat the prior comparison",
            query.answer_style("give some more brief about muralidhar"),
        )

    def test_explicit_entity_question_does_not_inherit_old_query(self) -> None:
        history = [{"role": "user", "content": "Who is a different person?"}]
        self.assertEqual(
            query.retrieval_terms("Who is Muralidhar Badisa?", history),
            "Who is Muralidhar Badisa?",
        )

    def test_follow_up_name_patterns_ignore_generic_expansion_words(self) -> None:
        self.assertEqual(
            query.named_source_patterns("give some more brief about muralidhar"),
            ["%muralidhar%"],
        )

    def test_named_question_restricts_context_to_the_matching_resume(self) -> None:
        documents = [
            Document(
                page_content="Muralidhar Badisa built web applications.",
                metadata={"source": "/docs/MURALIDHAR_BADISA.pdf"},
            ),
            Document(
                page_content="Jagadeesh is also a software developer.",
                metadata={"source": "/docs/JAGADEESH.pdf"},
            ),
        ]

        selected = query.restrict_to_named_source(
            "Who is Muralidhar and how is he suitable for full stack web development?",
            documents,
        )

        self.assertEqual(
            [document.metadata["source"] for document in selected],
            ["/docs/MURALIDHAR_BADISA.pdf"],
        )

    def test_comparison_retrieval_loads_both_named_resume_documents(self) -> None:
        question = (
            "Who is best fit for full-stack development, Muralidhar or Jagadeesh?"
        )
        matching_paths = [
            ("/docs/JAGADEESH_CHILAKABATHULA_C.pdf",),
            ("/docs/MURALIDHAR_BADISA_FA.pdf",),
        ]
        all_chunks = [
            (
                "/docs/JAGADEESH_CHILAKABATHULA_C.pdf",
                index,
                f"Jagadeesh resume chunk {index}",
                0.6,
            )
            for index in range(3)
        ] + [
            (
                "/docs/MURALIDHAR_BADISA_FA.pdf",
                index,
                f"Muralidhar resume chunk {index}",
                0.7,
            )
            for index in range(4)
        ]
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchall.side_effect = [
            matching_paths,
            all_chunks,
        ]
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.0] * EMBEDDING_DIMENSIONS

        with (
            patch("query.required_setting", return_value="postgres://test"),
            patch("query.ollama_embeddings", return_value=embeddings),
            patch("query.psycopg.connect", return_value=connection),
        ):
            documents = query.retrieve(question, 5, [])

        self.assertEqual(len(documents), 7)
        self.assertEqual(
            {document.metadata["source"] for document in documents},
            {
                "/docs/MURALIDHAR_BADISA_FA.pdf",
                "/docs/JAGADEESH_CHILAKABATHULA_C.pdf",
            },
        )
        sql, params = connection.execute.call_args.args
        self.assertIn("d.source_path = ANY(%s)", sql)
        self.assertEqual(
            set(params[1]),
            {
                "/docs/MURALIDHAR_BADISA_FA.pdf",
                "/docs/JAGADEESH_CHILAKABATHULA_C.pdf",
            },
        )

    def test_named_follow_up_loads_resume_chunks_not_keyword_false_matches(self) -> None:
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchall.side_effect = [
            [("/docs/MURALIDHAR_BADISA_FA.pdf",)],
            [
                ("/docs/MURALIDHAR_BADISA_FA.pdf", index, f"Resume chunk {index}", 0.8)
                for index in range(4)
            ],
        ]
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.0] * EMBEDDING_DIMENSIONS
        history = [
            {"role": "user", "content": "Who is Muralidhar and does he fit full-stack?"},
            {
                "role": "assistant",
                "content": "Muralidhar is a full-stack developer.",
                "sources": [
                    {"source": "/docs/MURALIDHAR_BADISA_FA.pdf", "chunk": 0, "similarity": 0.8},
                    {"source": "/docs/exam-ticket.pdf", "chunk": 1, "similarity": 0.4},
                ],
            },
        ]

        with (
            patch("query.required_setting", return_value="postgres://test"),
            patch("query.ollama_embeddings", return_value=embeddings),
            patch("query.psycopg.connect", return_value=connection),
        ):
            documents = query.retrieve("give some more brief about muralidhar", 5, history)

        self.assertEqual(len(documents), 4)
        self.assertEqual(
            {document.metadata["source"] for document in documents},
            {"/docs/MURALIDHAR_BADISA_FA.pdf"},
        )
        chunk_query = next(
            call
            for call in connection.execute.call_args_list
            if "d.source_path = ANY(%s)" in call.args[0]
        )
        self.assertEqual(
            chunk_query.args[1][1],
            ["/docs/MURALIDHAR_BADISA_FA.pdf"],
        )

    def test_retrieval_uses_keyword_and_vector_ranking_for_source_reference(self) -> None:
        history = [
            {
                "role": "assistant",
                "content": "Earlier answer",
                "sources": [
                    {"source": "/docs/other.pdf", "chunk": 1},
                    {"source": "/docs/resume.pdf", "chunk": 2},
                ],
            }
        ]
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchall.return_value = [
            ("/docs/resume.pdf", 1, "Experience and skills", 0.8)
        ]
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.0] * EMBEDDING_DIMENSIONS

        with (
            patch("query.required_setting", return_value="postgres://test"),
            patch("query.ollama_embeddings", return_value=embeddings),
            patch("query.psycopg.connect", return_value=connection),
        ):
            documents = query.retrieve(
                "According to source 2, explain his experience",
                5,
                history,
            )

        sql, params = connection.execute.call_args.args
        self.assertIn("vector_candidates", sql)
        self.assertIn("lexical_candidates", sql)
        self.assertIn("d.source_path = %s", sql)
        self.assertIn("/docs/resume.pdf", params)
        self.assertEqual(documents[0].page_content, "Experience and skills")

    def test_embeddings_keep_the_qwen_instruction_prefix(self) -> None:
        model = Mock()
        model.embed_documents.return_value = [[0.0] * EMBEDDING_DIMENSIONS]

        with patch("rag_utils.ollama_embeddings", return_value=model):
            embeddings = embed_texts(["chunk text"], prefix="search_document: ")

        self.assertEqual(len(embeddings), 1)
        model.embed_documents.assert_called_once_with(["search_document: chunk text"])

    def test_incomplete_non_streamed_generation_is_not_accepted(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "generation limit"):
            query.answer_text(
                AIMessage(
                    content="An incomplete answer",
                    response_metadata={"done_reason": "length"},
                )
            )

    def test_non_streamed_answer_hides_thinking_content(self) -> None:
        answer = query.answer_text(
            AIMessage(
                content="<think>private reasoning</think>The answer is grounded.",
                response_metadata={"done_reason": "stop"},
            )
        )

        self.assertEqual(answer, "The answer is grounded.")

    def test_thinking_filter_hides_thoughts_across_chunk_boundaries(self) -> None:
        answer_filter = query.ThinkingFilter()

        visible = "".join(
            answer_filter.feed(chunk)
            for chunk in (
                "Before <thi",
                "nk>internal analysis",
                "</think>After",
            )
        )
        visible += answer_filter.finish(completed=True)

        self.assertEqual(visible, "Before After")
        self.assertFalse(answer_filter.in_thinking)

    def test_thinking_filter_does_not_accept_unclosed_thought_as_answer(self) -> None:
        answer_filter = query.ThinkingFilter()

        self.assertEqual(answer_filter.feed("<think>unfinished analysis"), "")
        self.assertEqual(answer_filter.finish(completed=False), "")
        self.assertTrue(answer_filter.in_thinking)

    def test_thinking_filter_discards_unmarked_reasoning_before_closing_tag(self) -> None:
        answer_filter = query.ThinkingFilter()

        self.assertEqual(answer_filter.feed("internal analysis"), "")
        self.assertEqual(answer_filter.feed("</think>Final answer"), "Final answer")
        self.assertEqual(answer_filter.finish(completed=True), "")

    def test_thinking_filter_releases_plain_answer_when_generation_finishes(self) -> None:
        answer_filter = query.ThinkingFilter()

        self.assertEqual(answer_filter.feed("Direct answer"), "")
        self.assertEqual(answer_filter.finish(completed=True), "Direct answer")

    def test_complete_sentence_can_finish_at_generation_limit(self) -> None:
        self.assertTrue(query.has_complete_ending("A complete answer."))
        self.assertFalse(query.has_complete_ending("A cut off answer because"))
        self.assertTrue(
            query.has_complete_user_answer(
                "Muralidhar built frontend and backend applications using "
                "React, Django, Express, PostgreSQL, Docker, and CI/CD. "
                "These examples demonstrate experience across the full stack."
            )
        )

    def test_long_pasted_document_is_not_repeated_as_chat_history(self) -> None:
        history = [
            {"role": "user", "content": "Resume text " + ("details " * 100)},
            {"role": "user", "content": "Who is Muralidhar?"},
        ]
        self.assertEqual(query.format_history(history), "user: Who is Muralidhar?")

    def test_lcel_chain_passes_retrieved_context_to_the_chat_model(self) -> None:
        document = Document(
            page_content="Useful document content",
            metadata={"source": "guide.md", "chunk": 2, "similarity": 0.9},
        )
        prompts = []

        def capture_prompt(prompt):
            prompts.append(prompt)
            return AIMessage(content="Grounded answer")

        fake_chat_model = RunnableLambda(capture_prompt)
        with (
            patch("query.retrieve", return_value=[document]),
            patch("query.ChatOllama", return_value=fake_chat_model) as chat_constructor,
        ):
            result = query.create_rag_chain().invoke(
                {"question": "What is useful?", "top_k": 1}
            )

        self.assertEqual(result["answer"].content, "Grounded answer")
        self.assertEqual(result["documents"], [document])
        messages = prompts[0].to_messages()
        self.assertIn("never your reasoning or process", messages[0].content)
        self.assertIn("actual frontend, backend", messages[0].content)
        self.assertIn("Useful document content", messages[1].content)
        self.assertIn("What is useful?", messages[1].content)
        self.assertIn(
            "Document label for attribution only: guide.md",
            messages[1].content,
        )
        self.assertNotIn("Source [", messages[1].content)
        self.assertIn("Never include document labels", messages[0].content)
        chat_constructor.assert_called_once()
        self.assertFalse(chat_constructor.call_args.kwargs["reasoning"])
        self.assertEqual(chat_constructor.call_args.kwargs["num_predict"], 2048)
        self.assertEqual(chat_constructor.call_args.kwargs["keep_alive"], 300)

    def test_stream_emits_sources_tokens_and_completion_as_json_lines(self) -> None:
        document = Document(
            page_content="Useful document content",
            metadata={"source": "guide.md", "chunk": 2, "similarity": 0.9},
        )
        class FakePrompt:
            def invoke(self, _inputs):
                class PromptValue:
                    def to_messages(self):
                        return []

                return PromptValue()

        model = Mock()
        model.stream.return_value = iter(
            [
                AIMessageChunk(
                    content="Grounded answer",
                    response_metadata={"done_reason": "stop"},
                )
            ]
        )
        output = StringIO()
        with (
            patch(
                "query.retrieve_for_chain",
                return_value={"question": "Question", "documents": [document]},
            ),
            patch("query.create_answer_prompt", return_value=FakePrompt()),
            patch("query.create_chat_model", return_value=model),
            redirect_stdout(output),
        ):
            query.stream_result("Question", 1, [])

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]["type"], "sources")
        self.assertEqual(events[1], {"type": "token", "content": "Grounded answer"})
        self.assertEqual(events[2], {"type": "done"})


    def test_stream_continues_when_model_hits_token_limit(self) -> None:
        class FakePrompt:
            def invoke(self, _inputs):
                class PromptValue:
                    def to_messages(self):
                        return []

                return PromptValue()

        first_model = Mock()
        first_model.stream.return_value = iter(
            [
                AIMessageChunk(
                    content="<think>Internal analysis",
                    response_metadata={"done_reason": "length"},
                )
            ]
        )
        second_model = Mock()
        second_model.stream.return_value = iter(
            [
                AIMessageChunk(
                    content="<think>Continuing analysis</think>Final answer.",
                    response_metadata={"done_reason": "stop"},
                )
            ]
        )
        output = StringIO()
        with (
            patch(
                "query.retrieve_for_chain",
                return_value={"question": "Question", "documents": []},
            ),
            patch("query.create_answer_prompt", return_value=FakePrompt()),
            patch(
                "query.create_chat_model",
                side_effect=[first_model, second_model],
            ),
            redirect_stdout(output),
        ):
            query.stream_result("Question", 1, [])

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        tokens = "".join(
            event["content"] for event in events if event["type"] == "token"
        )
        self.assertEqual(tokens, "Final answer.")
        self.assertEqual(events[-1]["type"], "done")

    def test_stream_completes_when_answer_finishes_before_hidden_reasoning(self) -> None:
        class FakePrompt:
            def invoke(self, _inputs):
                class PromptValue:
                    def to_messages(self):
                        return []

                return PromptValue()

        model = Mock()
        answer = (
            "Muralidhar has experience with frontend and backend development, "
            "REST APIs, relational databases, containerized deployment, and "
            "full-stack projects. These documented skills align with the role."
        )
        model.stream.return_value = iter(
            [
                AIMessageChunk(
                    content="</think>" + answer,
                    response_metadata={"done_reason": "length"},
                )
            ]
        )
        output = StringIO()
        with (
            patch(
                "query.retrieve_for_chain",
                return_value={"question": "Question", "documents": []},
            ),
            patch("query.create_answer_prompt", return_value=FakePrompt()),
            patch("query.create_chat_model", return_value=model),
            redirect_stdout(output),
        ):
            query.stream_result("Question", 1, [])

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]["type"], "done")
        self.assertFalse(any(event["type"] == "error" for event in events))


if __name__ == "__main__":
    unittest.main()
