# mcp_sse_check.py
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

URL = "http://127.0.0.1:8001/sse"

async def main():
    async with sse_client(URL) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            tools = await sess.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            res = await sess.call_tool("query_rag", {"question": "ping", "top_k": 1})
            print("OK:", bool(res.content))
asyncio.run(main())
