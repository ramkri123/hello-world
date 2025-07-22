#!/usr/bin/env python3
"""
Privacy-Preserving NLP Feature Extractor for Consortium
Converts raw transaction data + emails into anonymous behavioral features
"""

import re
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PrivacyPreservingNLP:
    """Extract anonymous behavioral features from transaction data and emails"""
    
    def __init__(self):
        # Authority keywords for detection
        self.authority_keywords = [
            'ceo', 'president', 'executive', 'director', 'vp', 'vice president',
            'chief', 'manager', 'boss', 'supervisor', 'head', 'lead'
        ]
        
        # Urgency indicators
        self.urgency_patterns = [
            'urgent', 'asap', 'immediately', 'time sensitive', 'deadline',
            'before close', 'rush', 'emergency', 'critical', 'quickly',
            'hurry', 'soon as possible', 'right away', 'today', 'now'
        ]
        
        # Social engineering tactics
        self.manipulation_patterns = [
            'confidential', 'secret', 'don\'t tell', 'between us', 'private',
            'special opportunity', 'limited time', 'exclusive', 'favor',
            'help me', 'trust me', 'believe me', 'personally'
        ]
        
        # Business justification patterns
        self.business_patterns = [
            'acquisition', 'merger', 'new vendor', 'strategic partner',
            'new client', 'business opportunity', 'investment', 'contract',
            'deal', 'partnership', 'supplier', 'vendor payment'
        ]
        
        # Communication anomaly patterns
        self.anomaly_patterns = [
            'spelling_errors', 'grammar_issues', 'unusual_formatting',
            'external_email', 'reply_to_different', 'urgent_bypass'
        ]

    def extract_transaction_features(self, transaction_data: Dict) -> List[float]:
        """Extract anonymous features from transaction data"""
        features = []
        
        # Amount-based features (0-4)
        amount = transaction_data.get('amount', 0)
        sender_balance = transaction_data.get('sender_balance', 0)
        avg_daily_spending = transaction_data.get('avg_daily_spending', 1000)
        
        # Feature 0: Amount ratio to balance (realistic range)
        amount_ratio = min(amount / max(sender_balance, 100000), 1.0)
        # Add noise to make it realistic
        amount_ratio = max(0, min(1, amount_ratio + np.random.normal(0, 0.05)))
        features.append(amount_ratio)
        
        # Feature 1: Amount ratio to daily spending (realistic range)
        daily_ratio = min(amount / max(avg_daily_spending, 10000), 3.0) / 3.0
        daily_ratio = max(0, min(1, daily_ratio + np.random.normal(0, 0.05)))
        features.append(daily_ratio)
        
        # Feature 2: Large amount flag (gradual not binary)
        large_amount_flag = min(amount / 200000, 1.0)
        large_amount_flag = max(0, min(1, large_amount_flag + np.random.normal(0, 0.03)))
        features.append(large_amount_flag)
        
        # Feature 3: Round amount suspicion (softer detection)
        round_amount = 0.8 if amount % 1000 == 0 else np.random.beta(1.5, 8)
        features.append(round_amount)
        
        # Feature 4: Business hours flag (realistic distribution)
        hour = transaction_data.get('hour', 12)
        business_hours = 0.9 if 9 <= hour <= 17 else np.random.beta(1, 4)
        features.append(business_hours)
        
        # Timing features (5-9)
        day_of_week = transaction_data.get('day_of_week', 2)  # Monday=0
        is_holiday = transaction_data.get('is_holiday', False)
        
        # Feature 5: Weekend flag
        weekend_flag = 1.0 if day_of_week >= 5 else 0.0
        features.append(weekend_flag)
        
        # Feature 6: Holiday flag
        holiday_flag = 1.0 if is_holiday else 0.0
        features.append(holiday_flag)
        
        # Feature 7: Late in day flag
        late_day_flag = 1.0 if hour >= 16 else 0.0
        features.append(late_day_flag)
        
        # Feature 8: Friday afternoon risk
        friday_afternoon = 1.0 if day_of_week == 4 and hour >= 15 else 0.0
        features.append(friday_afternoon)
        
        # Feature 9: Off-hours risk
        off_hours = 1.0 if hour < 8 or hour > 18 or weekend_flag or holiday_flag else 0.0
        features.append(off_hours)
        
        return features

    def extract_email_features(self, email_content: str) -> List[float]:
        """Extract anonymous behavioral features from email content"""
        if not email_content:
            return [0.0] * 15  # Return 15 zero features if no email
            
        email_lower = email_content.lower()
        features = []
        
        # Authority impersonation features (10-12)
        authority_score = self._calculate_authority_score(email_lower)
        features.append(authority_score)
        
        # Executive bypass indication
        exec_bypass = 1.0 if any(word in email_lower for word in ['from ceo', 'from president', 'executive order']) else 0.0
        features.append(exec_bypass)
        
        # Authority urgency combination
        authority_urgency = min(authority_score + self._calculate_urgency_score(email_lower), 1.0)
        features.append(authority_urgency)
        
        # Urgency and pressure features (13-15)
        urgency_score = self._calculate_urgency_score(email_lower)
        features.append(urgency_score)
        
        # Timing pressure
        timing_pressure = self._calculate_timing_pressure(email_lower)
        features.append(timing_pressure)
        
        # Multiple urgency indicators
        urgency_count = sum(1 for pattern in self.urgency_patterns if pattern in email_lower)
        multiple_urgency = min(urgency_count / 5.0, 1.0)
        features.append(multiple_urgency)
        
        # Social engineering features (16-18)
        manipulation_score = self._calculate_manipulation_score(email_lower)
        features.append(manipulation_score)
        
        # Confidentiality claims
        confidentiality = 1.0 if any(word in email_lower for word in ['confidential', 'secret', 'private']) else 0.0
        features.append(confidentiality)
        
        # Trust/relationship exploitation
        trust_exploitation = 1.0 if any(phrase in email_lower for phrase in ['trust me', 'help me', 'favor']) else 0.0
        features.append(trust_exploitation)
        
        # Business justification features (19-21)
        business_score = self._calculate_business_score(email_lower)
        features.append(business_score)
        
        # New relationship claims
        new_relationship = 1.0 if any(phrase in email_lower for phrase in ['new vendor', 'new partner', 'new client']) else 0.0
        features.append(new_relationship)
        
        # Acquisition/merger language
        acquisition_language = 1.0 if any(word in email_lower for word in ['acquisition', 'merger', 'strategic']) else 0.0
        features.append(acquisition_language)
        
        # Communication anomaly features (22-24)
        communication_score = self._calculate_communication_anomalies(email_content)
        features.append(communication_score)
        
        # Grammar/spelling issues
        grammar_issues = self._detect_grammar_issues(email_content)
        features.append(grammar_issues)
        
        # External email indicators
        external_indicators = self._detect_external_indicators(email_content)
        features.append(external_indicators)
        
        return features

    def extract_account_features(self, sender_data: Dict, receiver_data: Dict) -> List[float]:
        """Extract anonymous account-based features"""
        features = []
        
        # Sender account features (25-29)
        sender_age_years = sender_data.get('account_age_years', 1)
        sender_risk_score = sender_data.get('risk_score', 0.1)
        sender_transaction_count = sender_data.get('transaction_count', 100)
        
        # Feature 25: Sender account age (normalized)
        sender_age_norm = min(sender_age_years / 10.0, 1.0)
        features.append(sender_age_norm)
        
        # Feature 26: Sender risk profile
        features.append(sender_risk_score)
        
        # Feature 27: Sender transaction frequency
        transaction_frequency = min(sender_transaction_count / 1000.0, 1.0)
        features.append(transaction_frequency)
        
        # Feature 28: Sender business type risk
        business_type = sender_data.get('business_type', 'individual')
        business_risk = 0.3 if business_type == 'business' else 0.1
        features.append(business_risk)
        
        # Feature 29: Sender geographic risk
        geographic_risk = sender_data.get('geographic_risk', 0.1)
        features.append(geographic_risk)
        
        # Receiver account features (30-34)
        receiver_age_years = receiver_data.get('account_age_years', 1)
        receiver_risk_score = receiver_data.get('risk_score', 0.1)
        
        # Feature 30: Receiver account age (normalized)
        receiver_age_norm = min(receiver_age_years / 10.0, 1.0)
        features.append(receiver_age_norm)
        
        # Feature 31: New account flag (< 30 days)
        new_account_flag = 1.0 if receiver_age_years < (30/365) else 0.0
        features.append(new_account_flag)
        
        # Feature 32: Receiver risk profile
        features.append(receiver_risk_score)
        
        # Feature 33: Cross-bank relationship
        cross_bank = 1.0 if sender_data.get('bank') != receiver_data.get('bank') else 0.0
        features.append(cross_bank)
        
        # Feature 34: Receiver verification status
        verification_score = receiver_data.get('verification_score', 0.5)
        features.append(verification_score)
        
        return features

    def convert_to_anonymous_features(self, transaction_data: Dict, email_content: str = "", 
                                    sender_data: Dict = None, receiver_data: Dict = None) -> List[float]:
        """Main function: Convert raw data to anonymous feature vector"""
        
        logger.info("🔄 NLP FEATURE EXTRACTION:")
        logger.info(f"   📧 Email length: {len(email_content) if email_content else 0} chars")
        logger.info(f"   💰 Transaction amount: ${transaction_data.get('amount', 0):,.2f}")
        
        # Extract features from each component
        transaction_features = self.extract_transaction_features(transaction_data)
        email_features = self.extract_email_features(email_content)
        
        # Use default account data if not provided
        if sender_data is None:
            sender_data = {'account_age_years': 2.0, 'risk_score': 0.1, 'transaction_count': 500, 
                          'business_type': 'business', 'geographic_risk': 0.2, 'bank': 'bank_A'}
        if receiver_data is None:
            receiver_data = {'account_age_years': 0.01, 'risk_score': 0.3, 'verification_score': 0.4, 'bank': 'bank_B'}
            
        account_features = self.extract_account_features(sender_data, receiver_data)
        
        # Combine all features
        all_features = transaction_features + email_features + account_features
        
        # Log key extracted features
        logger.info("   🔍 Key Behavioral Features Extracted:")
        if len(all_features) > 10:
            logger.info(f"      Authority Score: {all_features[10]:.3f}")
        if len(all_features) > 13:
            logger.info(f"      Urgency Score: {all_features[13]:.3f}")
        if len(all_features) > 16:
            logger.info(f"      Manipulation Score: {all_features[16]:.3f}")
        if len(all_features) > 31:
            logger.info(f"      New Account Flag: {all_features[31]:.3f}")
            
        logger.info(f"   📊 Total Features Generated: {len(all_features)}")
        
        return all_features

    def _calculate_authority_score(self, email_lower: str) -> float:
        """Calculate authority impersonation score"""
        score = 0.0
        for keyword in self.authority_keywords:
            if keyword in email_lower:
                score += 0.15
        return min(score, 1.0)

    def _calculate_urgency_score(self, email_lower: str) -> float:
        """Calculate urgency language score"""
        score = 0.0
        for pattern in self.urgency_patterns:
            if pattern in email_lower:
                score += 0.1
        return min(score, 1.0)

    def _calculate_timing_pressure(self, email_lower: str) -> float:
        """Calculate timing pressure score"""
        timing_phrases = ['before close', 'end of day', 'deadline', 'expires', 'by friday', 'today']
        score = sum(0.2 for phrase in timing_phrases if phrase in email_lower)
        return min(score, 1.0)

    def _calculate_manipulation_score(self, email_lower: str) -> float:
        """Calculate social engineering/manipulation score"""
        score = 0.0
        for pattern in self.manipulation_patterns:
            if pattern in email_lower:
                score += 0.15
        return min(score, 1.0)

    def _calculate_business_score(self, email_lower: str) -> float:
        """Calculate business justification score"""
        score = 0.0
        for pattern in self.business_patterns:
            if pattern in email_lower:
                score += 0.1
        return min(score, 1.0)

    def _calculate_communication_anomalies(self, email_content: str) -> float:
        """Detect communication anomalies"""
        score = 0.0
        
        # Check for excessive capitalization
        if sum(1 for c in email_content if c.isupper()) / max(len(email_content), 1) > 0.3:
            score += 0.3
            
        # Check for excessive punctuation
        punctuation_count = sum(1 for c in email_content if c in '!?')
        if punctuation_count > 5:
            score += 0.2
            
        return min(score, 1.0)

    def _detect_grammar_issues(self, email_content: str) -> float:
        """Detect grammar and spelling issues (simplified)"""
        issues = 0
        
        # Simple heuristics for grammar issues
        sentences = email_content.split('.')
        for sentence in sentences:
            # Check for missing capitalization
            if sentence.strip() and not sentence.strip()[0].isupper():
                issues += 1
                
        return min(issues / max(len(sentences), 1), 1.0)

    def _detect_external_indicators(self, email_content: str) -> float:
        """Detect external email indicators"""
        external_patterns = ['external', 'outside', 'gmail', 'yahoo', 'hotmail']
        score = sum(0.2 for pattern in external_patterns if pattern.lower() in email_content.lower())
        return min(score, 1.0)


def create_demo_bec_email():
    """Create a demo BEC email for testing"""
    return """
    Hi John,
    
    This is CEO Sarah Wilson. We have an urgent strategic acquisition opportunity 
    that requires immediate action. Please wire $485,000 to our new strategic 
    partner Global Tech Solutions for the acquisition deposit.
    
    This is highly confidential and time sensitive - we need to complete this 
    before market close Friday. Please process this immediately and don't discuss 
    with anyone else on the team.
    
    Thanks for your help with this critical transaction.
    
    Best regards,
    Sarah Wilson
    CEO
    """


def main():
    """Test the NLP feature extractor"""
    nlp = PrivacyPreservingNLP()
    
    # Test with BEC email
    bec_email = create_demo_bec_email()
    
    transaction_data = {
        'amount': 485000,
        'sender_balance': 2300000,
        'avg_daily_spending': 50000,
        'hour': 16,  # 4 PM
        'day_of_week': 4,  # Friday
        'is_holiday': False
    }
    
    sender_data = {
        'account_age_years': 6.0,
        'risk_score': 0.05,
        'transaction_count': 2000,
        'business_type': 'business',
        'geographic_risk': 0.1,
        'bank': 'bank_A'
    }
    
    receiver_data = {
        'account_age_years': 0.008,  # 3 days
        'risk_score': 0.8,
        'verification_score': 0.2,
        'bank': 'bank_B'
    }
    
    # Extract features
    features = nlp.convert_to_anonymous_features(
        transaction_data, bec_email, sender_data, receiver_data
    )
    
    print("\n🎯 EXTRACTED ANONYMOUS FEATURES:")
    print(f"Feature vector length: {len(features)}")
    print(f"Features: {[f'{f:.3f}' for f in features[:20]]}")
    
    # Verify key features
    print(f"\n📊 KEY BEHAVIORAL INDICATORS:")
    print(f"Authority Score: {features[10]:.3f}")
    print(f"Urgency Score: {features[13]:.3f}") 
    print(f"Manipulation Score: {features[16]:.3f}")
    print(f"New Account Flag: {features[31]:.3f}")
    print(f"Friday Afternoon Risk: {features[8]:.3f}")


if __name__ == "__main__":
    main()
