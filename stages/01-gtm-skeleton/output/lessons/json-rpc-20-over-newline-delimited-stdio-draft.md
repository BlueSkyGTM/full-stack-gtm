# Lesson: JSON-RPC 2.0 Over Newline-Delimited Stdio

---

## Beat 1: Hook — Why This Protocol Exists

The stdio transport is how local tool servers communicate with LLM hosts without opening network ports. Every MCP server your GTM stack runs through Clay or n8n speaks this protocol under the hood. If you can't read a newline-delimited JSON-RPC message stream, you can't debug why your enrichment tool silently fails.

---

## Beat 2: Concept — The Message Frame

JSON-RPC 2.0 defines three message types: requests (expect a response), notifications (fire-and-forget), and responses (result or error). The stdio transport frames each message as a single JSON object terminated by `\n`. No HTTP headers, no content-length, no WebSocket handshake. The newline *is* the protocol boundary. The host writes to the server's stdin; the server writes to stdout. Stdin/stdout form two unidirectional channels that together create bidirectional RPC.

---

## Beat 3: Demonstration — Round-Trip in a Single Process

Build a minimal JSON-RPC 2.0 responder that reads from stdin, parses the `{"jsonrpc":"2.0","method":"...","id":1}` envelope, dispatches on method, and writes a response to stdout. Show a second process that pipes requests in and reads responses back. Both processes are just `node` scripts printing to the terminal.

**Exercise (Easy):** Modify the responder to handle a second method and return a different result. Print the full exchange to confirm framing.

**Exercise (Medium):** Add a notification handler that prints to stderr without sending a response, proving notifications are one-way.

---

## Beat 4: Use It — Where This Shows Up in GTM Tooling

[CITATION NEEDED — concept: MCP server deployment in GTM enrichment workflows]

The MCP specification mandates JSON-RPC 2.0 over stdio as its default transport. When you configure a tool server for lead enrichment—say, a company-domain-to-employee-count lookup—the host process spawns your server as a child process and exchanges JSON-RPC messages over stdin/stdout. The `tools/call` method is what triggers your enrichment logic. If the response envelope is malformed or missing the `id` field, the host silently drops the result and your enrichment pipeline returns nothing. This is the transport layer behind every MCP-based integration in Zone 02 enrichment workflows.

**Exercise (Hard):** Implement a mock enrichment server that exposes a `lookup_company` method via JSON-RPC over stdio, responding with structured data. Pipe in a batch of three requests and capture the three responses.

---

## Beat 5: Ship It — Process Lifecycle and Edge Cases

A production stdio server must handle: malformed JSON without crashing (return a parse error response), unexpected EOF on stdin (clean shutdown), and concurrent buffered writes that could interleave if logging to stderr. The server must flush stdout after every response—Node.js `writeSync` or Python's `flush=True`—or the host will hang waiting for a response that's still in a buffer.

**Exercise (Medium):** Add malformed-input handling to your server. Send one valid request, one broken JSON line, and one more valid request. Confirm the server responds to both valid requests and returns a JSON-RPC parse error for the middle one, all on separate lines.

---

## Beat 6: Debug It — Silent Failures You Will Hit

The three failures you will encounter: (1) a response object with no `id` field, which the host discards silently; (2) a log line accidentally printed to stdout instead of stderr, which the host tries to parse as JSON-RPC and fails; (3) a missing trailing newline, which causes the host to buffer indefinitely waiting for the frame boundary. Reproduce each failure and observe the symptom—no error message, just silence.

**Exercise (Hard):** Intentionally introduce each of the three failure modes into your server. For each one, send a request and document the observed behavior (hang, crash, silent drop). Then fix each one and confirm recovery.