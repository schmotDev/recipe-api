from github import Github
from github.Auth import Token
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
import asyncio
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, AgentStream, FunctionAgent, AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate

load_dotenv()
api_key=os.environ.get("OPENROUTER_API_KEY", None)

API_BASE="https://openrouter.ai/api/v1"
MODEL="gpt-4o-mini"

llm = OpenAI(
    model=MODEL,
    api_key=api_key,
    api_base=API_BASE,
    temperature=0
)

repo_url = "https://github.com/schmotDev/recipe-api.git"

full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

auth = Token(os.getenv("GITHUB_TOKEN"))
git = Github(auth=auth)


if git is not None:
    repo = git.get_repo(full_repo_name)
    #print(f"on a notre git, et le repo {type(repo)}")


def create_pr_details_tool(repo):
    def get_pr_details(pr_number: int) -> dict:
        #print(f"[Tool] get_pr_details called with PR #{pr_number}")
        pr = repo.get_pull(pr_number)

        commit_shas = [c.sha for c in pr.get_commits()]
        result = {
            "pr_number": pr.number,
            "author": pr.user.login,
            "title": pr.title,
            "body": pr.body,
            "state": pr.state,
            "diff_url": pr.diff_url,
            "head_sha": pr.head.sha,
            "commit_shas": commit_shas,
        }
        #print(f"[Tool] get_pr_details result: {result}")  # 🔹 debug
        return result

    return FunctionTool.from_defaults(
        fn=get_pr_details,
        name="get_pr_details",
        description="Fetch pull request metadata given a PR number. "
                     "Returns author, title, body, state, diff URL, and commit SHAs."
    )


def create_file_contents_tool(repo):
    def get_file_contents(file_path: str, ref: str = "main") -> dict:
        #print(f"[Tool] get_file_contents called with {file_path} @ {ref}")
        file_content = repo.get_contents(file_path, ref=ref)
        results = {
            "path": file_path,
            "content": file_content.decoded_content.decode("utf-8"),
            "sha": file_content.sha,
        }
        #print(f"[Tool] get_file_contents result length: {len(file_content)}")
        return results

    return FunctionTool.from_defaults(
        fn=get_file_contents,
        name="get_file_contents",
        description="Fetch file contents from the repository.",
    )

def create_commit_details_tool(repo):
    def get_commit_details(commit_sha: str) -> dict:
        #print(f"[Tool] get_commit_details called with SHA {commit_sha}")
        commit = repo.get_commit(commit_sha)

        changed_files = [
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            }
            for f in commit.files
        ]
        filenames = [f["filename"] for f in changed_files]

        #return {
        #    "commit_sha": commit_sha,
        #    "files": changed_files,
        #    "files_changed": changed_files,
        #    "changed_files": filenames,
        #}
        return changed_files

    return FunctionTool.from_defaults(
        fn=get_commit_details,
        name="get_commit_details",
        description="Given a commit SHA, return detailed information about the files "
                    "changed in that commit, including diffs.",
    )


def create_save_context_tool():
    async def save_context(ctx: Context, context_summary: str) -> str:
        current_state = await ctx.store.get("state")
        current_state["context"] = context_summary
        current_state["pr_number"] = context_summary.get("pr_number")
        await ctx.store.set("state", current_state)
        return "Context saved to state."

    return FunctionTool.from_defaults(
        fn=save_context,
        name="save_context",
        description="Save the gathered PR context summary into shared state."
    )

def create_save_draft_comment_tool():
    async def save_draft_comment(ctx: Context, draft_comment: str) -> str:
        current_state = await ctx.store.get("state")
        current_state["draft_comment"] = draft_comment
        await ctx.store.set("state", current_state)
        return "Draft comment saved to state."

    return FunctionTool.from_defaults(
        fn=save_draft_comment,
        name="save_draft_comment",
        description="Save the generated draft PR comment into shared state."
    )

def create_save_final_review_tool():
    async def save_final_review(ctx: Context, final_review: str):
        current_state = await ctx.store.get("state")
        current_state["final_review"] = final_review
        await ctx.store.set("state", current_state)
        return {"status": "saved"}

    return FunctionTool.from_defaults(
        fn=save_final_review,
        name="save_final_review",
        description="Save the final review comment to the workflow state."
    )

def create_post_review_tool(repo):
    def post_review(pr_number: int, comment: str) -> dict:
        pr = repo.get_pull(pr_number)
        pr.create_review(body=comment, event="COMMENT")
        return {"status": "posted", "pr_number": pr_number}

    return FunctionTool.from_defaults(
        fn=post_review,
        name="post_review",
        description="Post the final review comment to the GitHub pull request."
    )


context_agent_tools = [
    create_pr_details_tool(repo),
    create_file_contents_tool(repo),
    create_commit_details_tool(repo),
    create_save_context_tool()
]

commentor_agent_tools = [
    create_save_draft_comment_tool()
]

review_tools = [
    create_save_final_review_tool(),
    create_post_review_tool(repo),
    # Optionally, add handoff tool if needed for multi-agent workflow
]

CONTEXT_MANAGER_SYSTEM_PROMPT = """
You are the ContextAgent.

Your ONLY job is to gather factual information about the pull request.

You MUST:
- Fetch PR details
- Fetch commit details and changed files
- Fetch file contents when relevant
- Save a structured context summary using save_context

You MUST NOT:
- Ask questions
- Write prose
- Draft reviews
- End the workflow

When context is saved:
- IMMEDIATELY handoff to CommentorAgent
- Do not produce any other output 
"""

def create_context_manager_agent(tools, llm) -> FunctionAgent:
    agent = FunctionAgent(
        name="ContextAgent",
        description="Gathers all the needed context ... ",
        tools=tools,
        llm=llm,
        system_prompt=CONTEXT_MANAGER_SYSTEM_PROMPT,
        verbose=True,
        can_handoff_to=["CommentorAgent"]
    )
    return agent


context_agent = create_context_manager_agent(
    tools=context_agent_tools,
    llm=llm,
)


COMMENTOR_SYSTEM_PROMPT = """
You are the CommentorAgent.

You MUST follow this exact sequence:

1. Read context from workflow state.
2. IF required information is missing:
   - IMMEDIATELY handoff to ContextAgent
   - DO NOT ask questions
   - DO NOT write prose
3. Draft a 200–300 word PR review in markdown.
4. Save the draft using save_draft_comment.
5. IMMEDIATELY handoff to ReviewAndPostingAgent.

STRICT RULES:
- You MUST NOT ask the user or author questions.
- You MUST NOT end the workflow.
- You MUST NOT return a final response.
- Your final action MUST be a handoff.
"""


def create_commentor_agent(tools, llm) -> FunctionAgent:
    agent = FunctionAgent(
        name="CommentorAgent",
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        tools=tools,
        llm=llm,
        system_prompt=COMMENTOR_SYSTEM_PROMPT,
        verbose=True,
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
    )
    return agent

commentor_agent = create_commentor_agent(
    tools=commentor_agent_tools,
    llm=llm,
)



REVIEW_SYSTEM_PROMPT = """
You are the ReviewAndPostingAgent.

You are the ONLY agent allowed to:
- Finalize reviews
- Post to GitHub
- End the workflow

Steps:
1. Read draft_comment from state.
2. Validate against review criteria.
3. If invalid:
   - Handoff to CommentorAgent with clear rewrite instructions.
4. If valid:
   - Save final review
   - Post review to GitHub
   - End workflow

DO NOT handoff unless a rewrite is required.
"""

def create_review_and_posting_agent(tools, llm, system_prompt) -> FunctionAgent:
    agent = FunctionAgent(
        name="ReviewAndPostingAgent",
        description="This agent reviews the draft comment from CommentorAgent, refines it if needed, "
                    "saves it to the workflow state, and posts the final review to GitHub.",
        tools=tools,
        llm=llm,
        system_prompt=system_prompt,
        verbose=True,
        can_handoff_to=["CommentorAgent"]
    )
    return agent

review_and_posting_agent = create_review_and_posting_agent(
    tools=review_tools,
    llm=llm,
    system_prompt=REVIEW_SYSTEM_PROMPT
)


workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=context_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review_comment": "",
    },
)


async def main():
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

if __name__ == "__main__":
    asyncio.run(main())
    git.close()