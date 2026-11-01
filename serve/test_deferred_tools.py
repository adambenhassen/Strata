"""Deferred tools: a client that declares ToolSearch (Claude Code with tool search on) holds tools it did not
declare and runs them by name. The model often calls one without loading it first, most of all after a compaction.
Returned as text (#3075), the call never reaches the client, so it can neither run it nor answer with an error, and
the model retries the same text. With ToolSearch declared, a call for any well-formed name is a call; without it,
the old rule stands.
    python -m unittest serve.test_deferred_tools -v
"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

def tool(name):
    return {"name": name, "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}

CALL = ("<tool_call>\n<function=TaskUpdate>\n<parameter=taskId>\n2\n</parameter>\n"
        "<parameter=status>\ncompleted\n</parameter>\n</function>\n</tool_call>")

def events(text, tools):
    p = OutputParser(thinking=False, tools=tools, stream_tools=True, declared_only=True)
    evs = p.feed(text) + p.finish("stop")
    return evs, p

class DeferredTools(unittest.TestCase):
    def test_undeclared_call_with_toolsearch_declared_is_a_call(self):
        evs, p = events(CALL, [tool("ToolSearch"), tool("Bash")])
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual([c.name for c in calls], ["TaskUpdate"])
        # no schema to type by: "2" reads as a number; the client checks it and, for a deferred tool, its error says
        # to load the tool and retry
        self.assertEqual(calls[0].arguments, {"taskId": 2, "status": "completed"})
        self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "")
        self.assertEqual(p.unread, 0)

    def test_streamed_in_pieces(self):
        p = OutputParser(thinking=False, tools=[tool("ToolSearch")], stream_tools=True, declared_only=True)
        evs = []
        for i in range(0, len(CALL), 7):
            evs += p.feed(CALL[i:i + 7])
        evs += p.finish("stop")
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["TaskUpdate"])

    def test_without_toolsearch_the_undeclared_call_stays_text(self):
        evs, p = events(CALL, [tool("Bash")])
        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])
        self.assertEqual(p.unread, 1)
        self.assertEqual(p.unread_names, ["TaskUpdate"])

    def test_a_name_that_is_no_tool_name_stays_text(self):
        bad = CALL.replace("<function=TaskUpdate>", "<function=Task/Update!>")
        evs, p = events(bad, [tool("ToolSearch")])
        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])

if __name__ == "__main__":
    unittest.main()
