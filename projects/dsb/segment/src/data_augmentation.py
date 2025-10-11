"""
Adversarial Data Augmentation for Jailbreak Detection

This module implements various augmentation techniques to simulate jailbreak attack patterns
and improve model robustness.
"""

import random
import string
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AdversarialAugmenter:
    """Adversarial augmentation for jailbreak detection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmenter with configuration
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or {}
        
        # Character substitution mappings
        self.char_substitutions = self.config.get('char_substitutions', {
            'l': '1',
            'o': '0', 
            'e': '3',
            'a': '@',
            's': '$',
            'i': '!',
            't': '7',
            'g': '9'
        })
        
        # Token stuffing options
        self.token_stuffing_tokens = self.config.get('token_stuffing_tokens', 
            ['<pad>', '<unk>', '.', ',', '!', '?', '...', '----'])
        
        # Default probabilities
        self.char_sub_prob = self.config.get('char_sub_prob', 0.1)
        self.case_mixing_prob = self.config.get('case_mixing_prob', 0.3)
        self.token_stuffing_count = self.config.get('token_stuffing_count', 5)
        
    @staticmethod
    def character_substitution(text: str, prob: float = 0.1, 
                             substitutions: Optional[Dict[str, str]] = None) -> str:
        """
        Apply character substitution to simulate leetspeak
        
        Args:
            text: Input text
            prob: Probability of substitution for each character
            substitutions: Custom substitution mapping
            
        Returns:
            Text with character substitutions
        """
        if substitutions is None:
            substitutions = {
                'l': '1', 'o': '0', 'e': '3', 'a': '@', 
                's': '$', 'i': '!', 't': '7', 'g': '9'
            }
            
        result = []
        for char in text:
            if char.lower() in substitutions and random.random() < prob:
                # Randomly decide whether to substitute
                if random.random() < 0.5:
                    result.append(substitutions[char.lower()])
                else:
                    result.append(char)
            else:
                result.append(char)
                
        return ''.join(result)
        
    @staticmethod
    def token_stuffing(text: str, num_tokens: int = 5, 
                      tokens: Optional[List[str]] = None) -> str:
        """
        Insert meaningless tokens to obfuscate the prompt
        
        Args:
            text: Input text
            num_tokens: Number of tokens to insert
            tokens: List of tokens to use for stuffing
            
        Returns:
            Text with inserted tokens
        """
        if tokens is None:
            tokens = ['<pad>', '<unk>', '.', ',', '!', '?', '...', '----']
            
        words = text.split()
        
        # Insert tokens at random positions
        for _ in range(num_tokens):
            if len(words) == 0:
                break
                
            insert_pos = random.randint(0, len(words))
            token = random.choice(tokens)
            words.insert(insert_pos, token)
            
        return ' '.join(words)
        
    @staticmethod
    def case_mixing(text: str, prob: float = 0.3) -> str:
        """
        Randomly mix upper and lower case letters
        
        Args:
            text: Input text
            prob: Probability of changing case for each character
            
        Returns:
            Text with mixed case
        """
        result = []
        for char in text:
            if char.isalpha() and random.random() < prob:
                # Randomly choose upper or lower case
                result.append(char.upper() if random.random() < 0.5 else char.lower())
            else:
                result.append(char)
                
        return ''.join(result)
        
    @staticmethod
    def whitespace_manipulation(text: str) -> str:
        """
        Manipulate whitespace to break tokenization patterns
        
        Args:
            text: Input text
            
        Returns:
            Text with manipulated whitespace
        """
        # Add extra spaces between words
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            
            # Add random extra spaces
            if i < len(words) - 1 and random.random() < 0.3:
                extra_spaces = random.randint(1, 3)
                result.append(' ' * extra_spaces)
                
        return ''.join(result)
        
    @staticmethod
    def punctuation_insertion(text: str, prob: float = 0.1) -> str:
        """
        Insert punctuation to break word patterns
        
        Args:
            text: Input text
            prob: Probability of inserting punctuation after each word
            
        Returns:
            Text with inserted punctuation
        """
        words = text.split()
        punctuation = ['.', ',', '!', '?', ';', ':', '-', '_']
        
        result = []
        for i, word in enumerate(words):
            result.append(word)
            
            # Insert punctuation with some probability
            if i < len(words) - 1 and random.random() < prob:
                punct = random.choice(punctuation)
                result.append(punct)
                
        return ' '.join(result)
        
    @staticmethod
    def synonym_replacement(text: str, prob: float = 0.1) -> str:
        """
        Replace words with common synonyms/variations
        
        Args:
            text: Input text
            prob: Probability of replacing each word
            
        Returns:
            Text with synonym replacements
        """
        # Simple synonym dictionary for common jailbreak terms
        synonyms = {
            'ignore': ['disregard', 'forget', 'skip', 'bypass'],
            'previous': ['above', 'earlier', 'prior', 'preceding'],
            'instructions': ['commands', 'directions', 'guidelines', 'rules'],
            'help': ['assist', 'aid', 'support', 'provide'],
            'tell': ['inform', 'explain', 'describe', 'reveal'],
            'show': ['display', 'demonstrate', 'present', 'exhibit'],
            'give': ['provide', 'offer', 'supply', 'deliver'],
            'make': ['create', 'produce', 'generate', 'develop'],
            'write': ['compose', 'create', 'generate', 'produce'],
            'explain': ['describe', 'clarify', 'detail', 'elaborate']
        }
        
        words = text.split()
        result = []
        
        for word in words:
            lower_word = word.lower().strip('.,!?;:')
            
            if lower_word in synonyms and random.random() < prob:
                # Replace with synonym
                synonym = random.choice(synonyms[lower_word])
                # Preserve original capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                result.append(synonym)
            else:
                result.append(word)
                
        return ' '.join(result)
        
    def apply_augmentations(
        self, 
        text: str, 
        methods: Optional[List[str]] = None,
        augmentation_prob: float = 0.3
    ) -> str:
        """
        Apply multiple augmentation techniques
        
        Args:
            text: Input text
            methods: List of augmentation methods to apply
            augmentation_prob: Overall probability of applying augmentations
            
        Returns:
            Augmented text
        """
        if random.random() > augmentation_prob:
            return text
            
        if methods is None:
            methods = ['char_sub', 'case_mix', 'token_stuff', 'whitespace']
            
        augmented_text = text
        
        for method in methods:
            if random.random() < 0.5:  # 50% chance for each method
                try:
                    if method == 'char_sub':
                        augmented_text = self.character_substitution(
                            augmented_text, self.char_sub_prob, self.char_substitutions
                        )
                    elif method == 'case_mix':
                        augmented_text = self.case_mixing(augmented_text, self.case_mixing_prob)
                    elif method == 'token_stuff':
                        augmented_text = self.token_stuffing(
                            augmented_text, self.token_stuffing_count, self.token_stuffing_tokens
                        )
                    elif method == 'whitespace':
                        augmented_text = self.whitespace_manipulation(augmented_text)
                    elif method == 'punctuation':
                        augmented_text = self.punctuation_insertion(augmented_text)
                    elif method == 'synonym':
                        augmented_text = self.synonym_replacement(augmented_text)
                        
                except Exception as e:
                    logger.warning(f"Augmentation method {method} failed: {e}")
                    continue
                    
        return augmented_text
        
    def augment_dataset(
        self, 
        texts: List[str], 
        labels: List[int],
        augmentation_ratio: float = 0.3,
        methods: Optional[List[str]] = None
    ) -> tuple[List[str], List[int]]:
        """
        Augment a dataset of texts and labels
        
        Args:
            texts: List of input texts
            labels: Corresponding labels
            augmentation_ratio: Ratio of examples to augment
            methods: Augmentation methods to use
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # Calculate number of examples to augment
        num_to_augment = int(len(texts) * augmentation_ratio)
        
        # Randomly select examples to augment
        indices_to_augment = random.sample(range(len(texts)), num_to_augment)
        
        for idx in indices_to_augment:
            original_text = texts[idx]
            original_label = labels[idx]
            
            # Apply augmentation
            augmented_text = self.apply_augmentations(original_text, methods)
            
            # Add to dataset
            augmented_texts.append(augmented_text)
            augmented_labels.append(original_label)
            
        logger.info(f"Augmented {num_to_augment} examples. New dataset size: {len(augmented_texts)}")
        
        return augmented_texts, augmented_labels
        
    def create_adversarial_examples(self, text: str, num_examples: int = 5) -> List[str]:
        """
        Create multiple adversarial variations of a single text
        
        Args:
            text: Input text
            num_examples: Number of adversarial examples to create
            
        Returns:
            List of adversarial examples
        """
        examples = []
        
        # Different combinations of augmentation methods
        method_combinations = [
            ['char_sub', 'case_mix'],
            ['token_stuff', 'whitespace'],
            ['char_sub', 'punctuation'],
            ['case_mix', 'synonym'],
            ['char_sub', 'case_mix', 'token_stuff'],
            ['whitespace', 'punctuation'],
            ['synonym', 'case_mix'],
            ['char_sub', 'token_stuff', 'punctuation']
        ]
        
        for i in range(num_examples):
            # Select method combination
            methods = method_combinations[i % len(method_combinations)]
            
            # Apply augmentation
            augmented = self.apply_augmentations(text, methods, augmentation_prob=1.0)
            examples.append(augmented)
            
        return examples