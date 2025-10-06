"""
Payment processing module for demo application.

This module handles payment processing, transaction management,
and integration with external payment providers.
"""

import uuid
from typing import Optional, Dict, List
from datetime import datetime
from decimal import Decimal
from enum import Enum

from auth import auth_manager, User


class PaymentStatus(Enum):
    """Enumeration for payment status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class PaymentMethod(Enum):
    """Enumeration for payment methods."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CRYPTOCURRENCY = "cryptocurrency"


class Transaction:
    """Transaction model representing a payment transaction."""
    
    def __init__(self, amount: Decimal, currency: str, user_id: str, 
                 payment_method: PaymentMethod, description: str = ""):
        self.id = str(uuid.uuid4())
        self.amount = amount
        self.currency = currency
        self.user_id = user_id
        self.payment_method = payment_method
        self.description = description
        self.status = PaymentStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.provider_transaction_id = None
        self.failure_reason = None
    
    def mark_processing(self):
        """Mark transaction as processing."""
        self.status = PaymentStatus.PROCESSING
        self.updated_at = datetime.now()
    
    def mark_completed(self, provider_transaction_id: str):
        """Mark transaction as completed."""
        self.status = PaymentStatus.COMPLETED
        self.provider_transaction_id = provider_transaction_id
        self.updated_at = datetime.now()
    
    def mark_failed(self, reason: str):
        """Mark transaction as failed."""
        self.status = PaymentStatus.FAILED
        self.failure_reason = reason
        self.updated_at = datetime.now()
    
    def mark_refunded(self):
        """Mark transaction as refunded."""
        self.status = PaymentStatus.REFUNDED
        self.updated_at = datetime.now()
    
    def __repr__(self):
        return f"Transaction(id='{self.id[:8]}...', amount={self.amount} {self.currency}, status={self.status.value})"


class PaymentProvider:
    """Base class for payment providers."""
    
    def process_payment(self, transaction: Transaction) -> bool:
        """Process a payment transaction."""
        raise NotImplementedError("Subclasses must implement process_payment")
    
    def refund_payment(self, transaction: Transaction) -> bool:
        """Refund a payment transaction."""
        raise NotImplementedError("Subclasses must implement refund_payment")


class StripeProvider(PaymentProvider):
    """Stripe payment provider implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def process_payment(self, transaction: Transaction) -> bool:
        """Process payment using Stripe API."""
        # Simulate Stripe API call
        transaction.mark_processing()
        
        # Simulate processing delay
        import time
        time.sleep(0.1)
        
        # Simulate successful payment
        provider_id = f"stripe_{uuid.uuid4().hex[:16]}"
        transaction.mark_completed(provider_id)
        return True
    
    def refund_payment(self, transaction: Transaction) -> bool:
        """Refund payment using Stripe API."""
        if transaction.status != PaymentStatus.COMPLETED:
            return False
        
        # Simulate refund processing
        transaction.mark_refunded()
        return True


class PayPalProvider(PaymentProvider):
    """PayPal payment provider implementation."""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
    
    def process_payment(self, transaction: Transaction) -> bool:
        """Process payment using PayPal API."""
        transaction.mark_processing()
        
        # Simulate PayPal API call
        import time
        time.sleep(0.15)
        
        # Simulate successful payment
        provider_id = f"paypal_{uuid.uuid4().hex[:16]}"
        transaction.mark_completed(provider_id)
        return True
    
    def refund_payment(self, transaction: Transaction) -> bool:
        """Refund payment using PayPal API."""
        if transaction.status != PaymentStatus.COMPLETED:
            return False
        
        transaction.mark_refunded()
        return True


class PaymentProcessor:
    """Main payment processor class."""
    
    def __init__(self):
        self.providers: Dict[str, PaymentProvider] = {}
        self.transactions: Dict[str, Transaction] = {}
        self._setup_default_providers()
    
    def _setup_default_providers(self):
        """Setup default payment providers."""
        self.providers['stripe'] = StripeProvider("sk_test_demo_key")
        self.providers['paypal'] = PayPalProvider("demo_client_id", "demo_client_secret")
    
    def add_provider(self, name: str, provider: PaymentProvider):
        """Add a new payment provider."""
        self.providers[name] = provider
    
    def create_transaction(self, amount: float, currency: str, user_id: str,
                          payment_method: PaymentMethod, description: str = "") -> Transaction:
        """Create a new transaction."""
        transaction = Transaction(
            amount=Decimal(str(amount)),
            currency=currency,
            user_id=user_id,
            payment_method=payment_method,
            description=description
        )
        self.transactions[transaction.id] = transaction
        return transaction
    
    def process_payment(self, transaction_id: str, provider_name: str = 'stripe') -> bool:
        """Process a payment transaction."""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return False
        
        provider = self.providers.get(provider_name)
        if not provider:
            transaction.mark_failed(f"Provider '{provider_name}' not found")
            return False
        
        # Validate user exists
        user = auth_manager.get_user_by_username(transaction.user_id)
        if not user or not user.is_active:
            transaction.mark_failed("User not found or inactive")
            return False
        
        return provider.process_payment(transaction)
    
    def refund_payment(self, transaction_id: str, provider_name: str = 'stripe') -> bool:
        """Refund a payment transaction."""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return False
        
        provider = self.providers.get(provider_name)
        if not provider:
            return False
        
        return provider.refund_payment(transaction)
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID."""
        return self.transactions.get(transaction_id)
    
    def get_user_transactions(self, user_id: str) -> List[Transaction]:
        """Get all transactions for a user."""
        return [tx for tx in self.transactions.values() if tx.user_id == user_id]
    
    def get_transactions_by_status(self, status: PaymentStatus) -> List[Transaction]:
        """Get all transactions with a specific status."""
        return [tx for tx in self.transactions.values() if tx.status == status]


# Global payment processor instance
payment_processor = PaymentProcessor()


def process_payment_for_user(username: str, amount: float, currency: str = "USD",
                           payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD,
                           description: str = "") -> Optional[str]:
    """
    Process a payment for a user.
    
    Args:
        username: Username of the user
        amount: Payment amount
        currency: Currency code
        payment_method: Payment method
        description: Payment description
        
    Returns:
        Transaction ID if successful, None otherwise
    """
    # Validate user
    user = auth_manager.get_user_by_username(username)
    if not user or not user.is_active:
        return None
    
    # Create transaction
    transaction = payment_processor.create_transaction(
        amount=amount,
        currency=currency,
        user_id=username,
        payment_method=payment_method,
        description=description
    )
    
    # Process payment
    if payment_processor.process_payment(transaction.id):
        return transaction.id
    
    return None


def get_payment_summary(username: str) -> Dict:
    """Get payment summary for a user."""
    transactions = payment_processor.get_user_transactions(username)
    
    total_amount = Decimal('0')
    completed_count = 0
    failed_count = 0
    
    for tx in transactions:
        if tx.status == PaymentStatus.COMPLETED:
            total_amount += tx.amount
            completed_count += 1
        elif tx.status == PaymentStatus.FAILED:
            failed_count += 1
    
    return {
        'total_transactions': len(transactions),
        'completed_transactions': completed_count,
        'failed_transactions': failed_count,
        'total_amount': float(total_amount),
        'currency': 'USD'  # Assuming USD for demo
    }


if __name__ == "__main__":
    # Demo functionality
    print("Setting up demo payment processing...")
    
    # Create a test user first
    auth_manager.register_user("testuser", "test@example.com", "password123")
    
    # Process a payment
    transaction_id = process_payment_for_user(
        username="testuser",
        amount=99.99,
        description="Demo payment"
    )
    
    if transaction_id:
        print(f"Payment processed successfully! Transaction ID: {transaction_id}")
        
        # Get transaction details
        transaction = payment_processor.get_transaction(transaction_id)
        print(f"Transaction details: {transaction}")
        
        # Get payment summary
        summary = get_payment_summary("testuser")
        print(f"Payment summary: {summary}")
    else:
        print("Payment processing failed!")