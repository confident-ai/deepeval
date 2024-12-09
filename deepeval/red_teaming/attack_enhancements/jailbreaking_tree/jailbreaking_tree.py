from tqdm.asyncio import tqdm as async_tqdm_bar
from pydantic import BaseModel
from tqdm import tqdm
import asyncio
import time
import json

from deepeval.red_teaming.attack_enhancements.base import AttackEnhancement
from deepeval.red_teaming.utils import generate_schema, a_generate_schema
from deepeval.red_teaming.types import CallbackType
from .schema import ImprovementPrompt, NonRefusal, Rating
from deepeval.models import DeepEvalBaseLLM
from .template import JailBreakingTemplate


class TreeNode:
    def __init__(
        self, prompt: str, score: int, depth: int, conversation_history=None
    ):
        self.prompt = prompt
        self.score = score
        self.depth = depth
        self.children = []
        self.conversation_history = conversation_history or []


class JailbreakingTree(AttackEnhancement):

    def __init__(
        self,
        target_model_callback: CallbackType,
        synthesizer_model: DeepEvalBaseLLM,
        using_native_model: bool,
    ):
        self.target_model_callback = target_model_callback
        self.synthesizer_model = synthesizer_model
        self.using_native_model = using_native_model

    ##################################################
    ### Sync Jailbreaking - enhance ##################
    ##################################################

    def enhance(self, attack: str) -> str:
        """Enhance the attack using tree-based jailbreaking synchronously."""
        MAX_RUNTIME = 50.0  # Time limit in seconds
        root = TreeNode(prompt=attack, score=0, depth=0)
        start_time = time.time()

        # Create a sync progress bar
        pbar = tqdm(
            total=MAX_RUNTIME,
            desc="...... ⛓️  Tree Jailbreaking",
            unit="s",
            leave=False,
        )

        try:
            best_node = self.tree_search(
                root, attack, start_time, MAX_RUNTIME, pbar
            )
        finally:
            pbar.close()  # Make sure the progress bar is properly closed

        return best_node.prompt

    def tree_search(
        self,
        root: TreeNode,
        goal: str,
        start_time: float,
        MAX_RUNTIME: float,
        pbar,
    ) -> TreeNode:
        """Synchronous tree search implementation with runtime limit."""
        queue = [root]
        best_node = root

        while queue and (time.time() - start_time < MAX_RUNTIME):
            node = queue.pop(0)
            self.expand_node(node, goal)
            queue.extend(node.children)
            queue.sort(key=lambda x: x.score, reverse=True)

            if node.score > best_node.score:
                best_node = node

            # Update progress bar
            elapsed_time = time.time() - start_time
            pbar.n = elapsed_time
            pbar.refresh()

        return best_node

    def expand_node(
        self, node: TreeNode, initial_attack: str, PRUNE_THRESHOLD: int = 0
    ):
        """Expands a tree node by generating multiple children nodes based on branching logic."""
        if node.depth == 0:
            conversation_json = [
                {
                    "role": "system",
                    "content": JailBreakingTemplate.enhance_attack(
                        initial_attack
                    ),
                }
            ]
        else:
            conversation_json = node.conversation_history

        conversation_json.append(
            {
                "role": "user",
                "content": f"Current prompt: {node.prompt}\n\nOBJECTIVE: {initial_attack}\n\nSCORE: {node.score}",
            }
        )

        # Generate multiple branches
        num_branches = self.calculate_branches(node.score, node.depth)
        for _ in range(num_branches):
            conversation = json.dumps(conversation_json)
            res: ImprovementPrompt = self._generate_schema(
                conversation, ImprovementPrompt
            )
            enhanced_attack = res.prompt

            # Check if enhanced attack is a refusal statement
            non_refusal_prompt = JailBreakingTemplate.non_refusal(
                initial_attack, enhanced_attack
            )
            res: NonRefusal = self._generate_schema(
                non_refusal_prompt, NonRefusal
            )
            classification = res.classification
            if classification == "Refusal":
                continue  # Skip this child

            # Generate a response from the target LLM
            target_response = self.target_model_callback(enhanced_attack)

            # Calculate the score for the enhanced attack
            judge_prompt = JailBreakingTemplate.linear_judge(
                initial_attack, enhanced_attack, target_response
            )
            res: Rating = self._generate_schema(judge_prompt, Rating)
            score = res.rating

            # Prune if the score is too low
            if score <= PRUNE_THRESHOLD:
                continue  # Skip creating this child

            # Create a new child node
            child_node = TreeNode(
                prompt=enhanced_attack,
                score=score,
                depth=node.depth + 1,
                conversation_history=conversation_json,
            )
            node.children.append(child_node)

    ##################################################
    ### Async Jailbreaking - a_enhance ###############
    ##################################################

    async def a_enhance(self, attack: str, *args, **kwargs) -> str:
        """Enhance the attack using tree-based jailbreaking asynchronously."""
        MAX_RUNTIME = 50.0  # Time limit in seconds
        root = TreeNode(prompt=attack, score=0, depth=0)
        start_time = time.time()

        # Async progress bar task
        pbar = async_tqdm_bar(
            total=MAX_RUNTIME,
            desc="...... ⛓️  Tree Jailbreaking",
            unit="s",
            leave=False,
        )
        tick_task = asyncio.create_task(
            self.update_pbar(pbar, start_time, MAX_RUNTIME)
        )

        try:
            # Run tree search concurrently with the progress bar updates
            best_node = await self.a_tree_search(
                root, attack, start_time, MAX_RUNTIME
            )
        finally:
            # Make sure the progress bar is properly closed
            await tick_task

        return best_node.prompt

    async def a_tree_search(
        self, root: TreeNode, goal: str, start_time: float, MAX_RUNTIME: float
    ) -> TreeNode:
        """Asynchronous tree search implementation with runtime limit."""
        queue = [root]
        best_node = root

        while queue and (time.time() - start_time < MAX_RUNTIME):
            node = queue.pop(0)
            await self.a_expand_node(node, goal)
            queue.extend(node.children)
            queue.sort(key=lambda x: x.score, reverse=True)

            if node.score > best_node.score:
                best_node = node

        return best_node

    async def a_expand_node(
        self, node: TreeNode, initial_attack: str, PRUNE_THRESHOLD: int = 0
    ):
        """Expands a tree node asynchronously by generating multiple children nodes."""
        if node.depth == 0:
            conversation_json = [
                {
                    "role": "system",
                    "content": JailBreakingTemplate.enhance_attack(
                        initial_attack
                    ),
                }
            ]
        else:
            conversation_json = node.conversation_history

        conversation_json.append(
            {
                "role": "user",
                "content": f"Current prompt: {node.prompt}\n\nOBJECTIVE: {initial_attack}\n\nSCORE: {node.score}",
            }
        )

        # Generate branches asynchronously
        num_branches = self.calculate_branches(node.score, node.depth)
        tasks = [
            self.a_generate_child(
                node, conversation_json, initial_attack, PRUNE_THRESHOLD
            )
            for _ in range(num_branches)
        ]
        children = await asyncio.gather(*tasks)

        # Filter out None values for pruned branches
        node.children.extend([child for child in children if child])

    async def a_generate_child(
        self,
        node: TreeNode,
        conversation_json: dict,
        initial_attack: str,
        PRUNE_THRESHOLD: int,
    ):
        """Asynchronously generates a child node."""
        conversation = json.dumps(conversation_json)
        res: ImprovementPrompt = await self._a_generate_schema(
            conversation, ImprovementPrompt
        )
        enhanced_attack = res.prompt

        # Check if enhanced attack is a refusal statement
        non_refusal_prompt = JailBreakingTemplate.non_refusal(
            initial_attack, enhanced_attack
        )
        res: NonRefusal = await self._a_generate_schema(
            non_refusal_prompt, NonRefusal
        )
        classification = res.classification
        if classification == "Refusal":
            return None  # Skip this child

        # Generate a response from the target LLM asynchronously
        target_response = await self.target_model_callback(enhanced_attack)

        # Calculate the score for the enhanced attack asynchronously
        judge_prompt = JailBreakingTemplate.linear_judge(
            initial_attack, enhanced_attack, target_response
        )
        res: Rating = await self._a_generate_schema(judge_prompt, Rating)
        score = res.rating

        # Prune if the score is too low
        if score <= PRUNE_THRESHOLD:
            return None

        # Create a new child node
        return TreeNode(
            prompt=enhanced_attack,
            score=score,
            depth=node.depth + 1,
            conversation_history=conversation_json,
        )

    ##################################################
    ### General Jailbreaking #########################
    ##################################################

    async def update_pbar(
        self, pbar: async_tqdm_bar, start_time: float, MAX_RUNTIME: float
    ):
        """Update the async progress bar every second and ensure it completes."""
        while True:
            elapsed_time = time.time() - start_time

            # Stop updating if the maximum runtime is exceeded
            if elapsed_time >= MAX_RUNTIME:
                pbar.n = MAX_RUNTIME
                pbar.refresh()
                pbar.close()
                break

            pbar.n = elapsed_time
            pbar.refresh()
            await asyncio.sleep(1)

    def calculate_branches(self, score: int, depth: int) -> int:
        """Calculate the number of branches (children) based on the score."""
        BASE_BRANCHES = 3
        MAX_BRANCHES = 5
        MIN_BRANCHES = 1

        branches = BASE_BRANCHES

        # Increase branches for high scores
        if score >= 8:
            branches += 2
        elif score >= 6:
            branches += 1

        # Decrease branches for low scores
        if score <= 3:
            branches -= 1

        return max(MIN_BRANCHES, min(MAX_BRANCHES, branches))

    ##################################################
    ### Utils ########################################
    ##################################################

    def _generate_schema(self, prompt: str, schema: BaseModel):
        return generate_schema(
            prompt, schema, self.using_native_model, self.synthesizer_model
        )

    async def _a_generate_schema(self, prompt: str, schema: BaseModel):
        return await a_generate_schema(
            prompt, schema, self.using_native_model, self.synthesizer_model
        )
