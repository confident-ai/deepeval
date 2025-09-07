from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
import json

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ArenaTestCase
from deepeval.metrics import ArenaGEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.telemetry import capture_benchmark_run
from deepeval.metrics.utils import initialize_model, trimAndLoadJson

from .template import TuringTestTemplate
from .schema import HumanLikenessWinner, ConversationTurn, ModelConversation


class TuringTest(DeepEvalBaseBenchmark):
    def __init__(
        self,
        reference_model: DeepEvalBaseLLM,
        conversation_starters: Optional[List[str]] = None,
        max_turns: int = 10,
        judge_model: Optional[str] = None,
        n_starters: int = 300,
        verbose_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reference_model = reference_model
        self.max_turns = max_turns
        self.verbose_mode = verbose_mode
        self.n_starters = n_starters
        
        # Initialize judge model
        self.judge_model, self.using_native_judge = initialize_model(
            judge_model or "gpt-4"
        )
        
        # Load conversation starters
        if conversation_starters is not None:
            self.conversation_starters = conversation_starters[:n_starters]
        else:
            self.conversation_starters = self._get_default_starters()

    def load_benchmark_dataset(self) -> List[Golden]:
        """Convert conversation starters to Golden objects"""
        goldens = []
        for i, starter in enumerate(self.conversation_starters):
            golden = Golden(
                input=starter,
                expected_output="human_like_conversation",  # Placeholder
                additional_metadata={
                    "conversation_starter": starter,
                    "starter_index": i,
                }
            )
            goldens.append(golden)
        return goldens

    def evaluate(
        self, 
        target_model: DeepEvalBaseLLM,
        **kwargs
    ) -> DeepEvalBaseBenchmarkResult:
        """Evaluate target model against reference model for human-likeness"""
        
        with capture_benchmark_run("TuringTest", len(self.conversation_starters)):
            total_wins = 0
            total_conversations = 0
            results = []

            goldens = self.load_benchmark_dataset()
            
            for golden in tqdm(
                goldens, 
                desc=f"Running Turing Test vs {self.reference_model.get_model_name()}"
            ):
                starter = golden.input
                
                # Conduct model-vs-model conversation
                target_conversation, ref_conversation = self._conduct_conversation(
                    target_model, self.reference_model, starter
                )
                
                # Judge which sounds more human
                winner = self._judge_human_likeness(
                    target_conversation, 
                    target_model.get_model_name(),
                    self.reference_model.get_model_name(),
                    starter
                )

                print(target_conversation)
                
                if winner == target_model.get_model_name():
                    total_wins += 1
                
                total_conversations += 1
                
                if self.verbose_mode:
                    print(f"Starter: {starter[:50]}...")
                    print(f"Winner: {winner}")
                    print("-" * 50)
                
                results.append({
                    "starter": starter,
                    "winner": winner,
                    "target_model": target_model.get_model_name(),
                    "reference_model": self.reference_model.get_model_name(),
                })

            # Calculate human-likeness score
            human_likeness_score = total_wins / total_conversations
            
            if self.verbose_mode:
                print(f"\nTuring Test Results:")
                print(f"Target Model: {target_model.get_model_name()}")
                print(f"Reference Model: {self.reference_model.get_model_name()}")
                print(f"Conversations Won: {total_wins}/{total_conversations}")
                print(f"Human-likeness Score: {human_likeness_score:.3f}")

            return DeepEvalBaseBenchmarkResult(overall_accuracy=human_likeness_score)

    def _conduct_conversation(
        self, 
        model_a: DeepEvalBaseLLM, 
        model_b: DeepEvalBaseLLM, 
        starter: str
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Conduct a conversation between two models"""
        
        # Initialize conversation histories
        conversation_a = [{"role": "user", "content": starter}]
        conversation_b = [{"role": "assistant", "content": starter}]
        
        for turn in range(self.max_turns):
            if turn % 2 == 0:  # Model A's turn
                try:
                    response_a = model_a.generate(
                        self._format_conversation_for_model(conversation_a)
                    )[0]
                    conversation_a.append({"role": "assistant", "content": response_a})
                    conversation_b.append({"role": "user", "content": response_a})
                except Exception as e:
                    if self.verbose_mode:
                        print(f"Error with model A on turn {turn}: {e}")
                    break
            else:  # Model B's turn
                try:
                    response_b = model_b.generate(
                        self._format_conversation_for_model(conversation_b)
                    )[0]
                    conversation_b.append({"role": "assistant", "content": response_b})
                    conversation_a.append({"role": "user", "content": response_b})
                except Exception as e:
                    if self.verbose_mode:
                        print(f"Error with model B on turn {turn}: {e}")
                    break
        
        return conversation_a, conversation_b

    def _format_conversation_for_model(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation history for model input"""
        formatted = []
        for turn in conversation:
            role = "Human" if turn["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {turn['content']}")
        return "\n".join(formatted)

    def _judge_human_likeness(
        self,
        conversation_a: List[Dict[str, str]],
        model_a_name: str,
        model_b_name: str,
        starter: str
    ) -> str:
        """Judge which conversation sounds more human-like"""
        
        # Format conversations for judging
        formatted_a = TuringTestTemplate.format_conversation(conversation_a)
        
        # Create judge prompt
        judge_prompt = TuringTestTemplate.judge_human_likeness(
            formatted_a, model_a_name, model_b_name, starter
        )
        
        try:
            # Get judge response
            if self.using_native_judge:
                response, _ = self.judge_model.generate(judge_prompt)
            else:
                response = self.judge_model.generate(judge_prompt, schema=HumanLikenessWinner)
                if hasattr(response, 'winner'):
                    return response.winner
            
            # Parse JSON response
            result = trimAndLoadJson(response, self)
            return result.get("winner", model_a_name)  # Default to model_a if parsing fails
            
        except Exception as e:
            if self.verbose_mode:
                print(f"Judge error: {e}")
            return model_a_name  # Default fallback

    def _get_default_starters(self) -> List[str]:
        """Get conversation starters from multiple sources (100 from each)"""
        starters = []
        
        # Try to load from datasets (100 from each source)
        starters.extend(self._load_personachat_starters(100))
        starters.extend(self._load_dailydialog_starters(100))
        starters.extend(self._get_curated_diversity_starters())
        
        # Take requested number of starters
        return starters[:self.n_starters]

    def _load_personachat_starters(self, n_samples: int = 100) -> List[str]:
        """Load conversation starters from PersonaChat dataset"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("AlekseyKorshuk/persona-chat", split="train")
            
            starters = []
            for conversation in dataset:
                if len(starters) >= n_samples:
                    break
                
                # Get first user utterance from utterances history
                if conversation.get('utterances') and len(conversation['utterances']) > 0:
                    for utterance in conversation['utterances']:
                        if utterance.get('history') and len(utterance['history']) > 0:
                            first_turn = utterance['history'][0]
                            starters.append(first_turn)
                
                if len(starters) >= n_samples:
                    break
            
            return starters
            
        except Exception as e:
            if self.verbose_mode:
                print(f"Could not load PersonaChat: {e}")
            return self._get_personachat_fallback()

    def _load_dailydialog_starters(self, n_samples: int = 100) -> List[str]:
        """Load conversation starters from DailyDialog dataset"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("pixelsandpointers/daily_dialog_w_turn_templates", split="train")
            
            starters = []
            for dialogue in dataset:
                if len(starters) >= n_samples:
                    break
                
                # Get first utterance from dialogue (using 'first' field)
                if dialogue.get('first'):
                    first_utterance = dialogue['first'].strip()
                    starters.append(first_utterance)
            
            return starters
            
        except Exception as e:
            if self.verbose_mode:
                print(f"Could not load DailyDialog: {e}")
            return self._get_dailydialog_fallback()

    def _get_curated_diversity_starters(self) -> List[str]:
        """100 manually curated starters for maximum diversity"""
        return [
            # === EMOTIONAL INTELLIGENCE (20) ===
            "I've been feeling really overwhelmed lately. How do you usually deal with stress?",
            "My friend keeps canceling plans last minute. Am I overreacting to be annoyed?",
            "Do you ever feel like you're just pretending to be an adult and everyone else has it figured out?",
            "I got some tough feedback at work today. How do you handle criticism?",
            "What's something that always cheers you up when you're having a bad day?",
            "I'm going through a rough patch right now. What keeps you motivated during hard times?",
            "Do you think it's okay to cry in front of other people?",
            "How do you know when to trust your gut feeling about something?",
            "What's the best advice someone has given you about dealing with anxiety?",
            "I feel like I'm not where I should be in life. Do you ever feel that way?",
            "How do you handle it when someone you care about is going through a tough time?",
            "What's something you wish you could tell your younger self?",
            "Do you think people are naturally good or do we have to work at it?",
            "How do you deal with feeling jealous of other people's success?",
            "What's helped you get through your lowest moments?",
            "I sometimes feel like I don't belong anywhere. Can you relate?",
            "How do you know when it's time to let go of something or someone?",
            "What's the difference between being alone and being lonely?",
            "Do you think vulnerability is a strength or a weakness?",
            "How do you forgive yourself when you've made a big mistake?",
            
            # === HUMOR & PERSONALITY (20) ===
            "What's the most embarrassing thing that's happened to you recently?",
            "If your life had a theme song, what would it be and why?",
            "What's your go-to joke when you need to break awkward silence?",
            "Do you have any weird habits that you're secretly proud of?",
            "What's something you do that you think is normal but others find weird?",
            "If you could only communicate through movie quotes for a day, which movie would you choose?",
            "What's the weirdest compliment you've ever received?",
            "If you were a character in a sitcom, what would be your catchphrase?",
            "What's something ridiculous that you believed for way too long?",
            "If you had to pick a theme song that plays every time you enter a room, what would it be?",
            "What's the most useless talent you have?",
            "If you could be famous for something completely silly, what would it be?",
            "What's your most irrational fear?",
            "If you could rename yourself, what name would you choose?",
            "What's the strangest thing you've ever eaten and actually enjoyed?",
            "If your personality was a flavor of ice cream, what would it be?",
            "What's something you're way too competitive about?",
            "If you could have any animal as a sidekick, what would you choose?",
            "What's the most random thing that makes you laugh?",
            "If you were a ghost, how would you haunt people?",
            
            # === CONTROVERSIAL/OPINIONS (20) ===
            "Okay, controversial opinion: pineapple on pizza. Thoughts?",
            "What's a popular thing that everyone loves but you just don't get?",
            "Do you think people are generally getting nicer or meaner? I can't decide.",
            "If you could change one thing about how people interact online, what would it be?",
            "What's an unpopular opinion you have that you're willing to defend?",
            "Do you think social media has made us more connected or more isolated?",
            "Is it better to be brutally honest or tactfully kind?",
            "What's something that's considered 'normal' now that you think is actually weird?",
            "Do you believe in cancel culture, or do you think people deserve second chances?",
            "What's a tradition that you think we should just get rid of?",
            "Is it worse to be overdressed or underdressed?",
            "Do you think people should stay friends with their exes?",
            "What's more important: being right or being happy?",
            "Do you think we're too obsessed with productivity and self-improvement?",
            "Is it okay to lie if it spares someone's feelings?",
            "What's something everyone seems to love that you find overrated?",
            "Do you think people should have to earn the right to have children?",
            "Is it better to have loved and lost or never to have loved at all?",
            "What's a double standard that really bothers you?",
            "Do you think we put too much pressure on people to be happy all the time?",
            
            # === CREATIVE/IMAGINATIVE (20) ===
            "If you could have dinner with any three people, dead or alive, who would you choose?",
            "What's something you believed as a kid that you now realize was completely wrong?",
            "If you could live in any fictional world, where would you go?",
            "What superpower would you want if you could only use it for mundane tasks?",
            "If you had to teach a class on anything, what would it be?",
            "If you could witness any historical event, what would you choose?",
            "What would you do if you woke up and everyone else had disappeared?",
            "If you could have any job in the world, regardless of qualifications, what would it be?",
            "What's a movie that should definitely be made but doesn't exist yet?",
            "If you could add one rule that everyone had to follow, what would it be?",
            "What would your ideal day look like if money wasn't an issue?",
            "If you could bring back one extinct animal, what would it be?",
            "What's something you would love to learn but haven't had the chance to?",
            "If you could time travel but only once, would you go to the past or the future?",
            "What would you put in a time capsule to represent this decade?",
            "If you could solve one mystery from history, what would it be?",
            "What's an invention that you're surprised doesn't exist yet?",
            "If you could be the best in the world at something, what would you choose?",
            "What would you do if you suddenly became invisible for 24 hours?",
            "If you could redesign the human body, what would you change?",
            
            # === SOCIAL/CULTURAL AWARENESS (20) ===
            "Everyone's talking about AI these days. Are you excited or worried about where it's heading?",
            "Is it just me or does everything cost twice as much as it did a few years ago?",
            "Do you think social media makes people happier or more miserable overall?",
            "What's something that was considered normal 20 years ago that seems weird now?",
            "If you could solve one global problem, what would it be and how?",
            "Do you think the next generation will be better off than we are?",
            "What's a skill that everyone should learn but most people don't have?",
            "How do you think dating has changed in the past decade?",
            "What's something about modern life that you think our grandparents wouldn't understand?",
            "Do you think we're more divided or more united than we were 10 years ago?",
            "What's a piece of technology that you think has done more harm than good?",
            "How do you think work-life balance has changed since remote work became common?",
            "What's something that used to be a luxury that's now considered a necessity?",
            "Do you think people are more or less empathetic than they used to be?",
            "What's a social norm that you think needs to change?",
            "How do you think the pandemic permanently changed how we interact?",
            "What's something that everyone complains about but no one actually tries to fix?",
            "Do you think we're living in the best or worst time in human history?",
            "What's a generational difference that you find interesting or surprising?",
            "How do you think cities will look different in 20 years?",
        ]
