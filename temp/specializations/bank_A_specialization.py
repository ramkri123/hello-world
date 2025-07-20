#!/usr/bin/env python3
"""
Bank A Specialization - Wire Transfer Fraud Detection
Custom business logic for Bank A's wire transfer expertise
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

def customize_node(node):
    """Customize the participant node with Bank A specific behavior"""
    logger.info("🔧 Applying Bank A specializations...")
    
    # Override inference processing for wire transfer specialization
    original_process_inference = node.process_inference
    
    def enhanced_wire_transfer_inference(features):
        """Enhanced inference with wire transfer specialization"""
        try:
            # Bank A focuses on wire transfer patterns
            # Features 0-4: Amount-related features
            # Features 5-9: Geographic features  
            # Features 10-14: Business legitimacy features
            
            amount_risk = analyze_amount_patterns(features[:5])
            geographic_risk = analyze_geographic_patterns(features[5:10])
            business_risk = analyze_business_legitimacy(features[10:15])
            
            # Weighted combination based on Bank A's expertise
            specialized_score = (
                0.4 * amount_risk +      # High weight on amounts
                0.3 * geographic_risk +  # Medium weight on geography
                0.3 * business_risk      # Medium weight on business factors
            )
            
            # Call original inference and combine
            original_result = original_process_inference(features)
            original_score = original_result.get('risk_score', 0.5)
            
            # Blend specialized and general scores
            final_score = 0.7 * specialized_score + 0.3 * original_score
            final_score = max(0.0, min(1.0, final_score))
            
            logger.info(f"🎯 Wire Transfer Analysis: specialized={specialized_score:.3f}, "
                       f"general={original_score:.3f}, final={final_score:.3f}")
            
            return {
                "risk_score": final_score,
                "confidence": 0.92,  # High confidence in wire transfer detection
                "specialized_components": {
                    "amount_risk": amount_risk,
                    "geographic_risk": geographic_risk, 
                    "business_risk": business_risk
                },
                "specialization": "wire_transfer_expert",
                "processed_at": original_result.get('processed_at'),
                "model_version": "1.0-wire-specialist"
            }
            
        except Exception as e:
            logger.error(f"❌ Wire transfer specialization error: {e}")
            return original_process_inference(features)
    
    # Replace the inference method
    node.process_inference = enhanced_wire_transfer_inference
    logger.info("✅ Bank A wire transfer specializations applied")

def analyze_amount_patterns(amount_features):
    """Analyze transaction amount patterns for wire transfer fraud"""
    try:
        # High amounts with unusual patterns are suspicious
        avg_amount = np.mean(amount_features)
        amount_variance = np.var(amount_features)
        
        # Wire transfers over certain thresholds are higher risk
        if avg_amount > 0.8:  # High amount transactions
            base_risk = 0.7
        elif avg_amount > 0.6:  # Medium-high amounts
            base_risk = 0.5  
        else:
            base_risk = 0.3
            
        # High variance in amounts suggests structured transactions
        variance_penalty = min(0.3, amount_variance * 2)
        
        return min(1.0, base_risk + variance_penalty)
        
    except Exception as e:
        logger.error(f"Amount analysis error: {e}")
        return 0.5

def analyze_geographic_patterns(geo_features):
    """Analyze geographic patterns for wire transfer fraud"""
    try:
        # Unusual geographic patterns increase risk
        geo_avg = np.mean(geo_features)
        geo_spread = np.max(geo_features) - np.min(geo_features)
        
        # High geographic spread suggests cross-border activity
        if geo_spread > 0.7:
            return min(1.0, 0.6 + geo_avg * 0.4)
        else:
            return geo_avg * 0.6
            
    except Exception as e:
        logger.error(f"Geographic analysis error: {e}")
        return 0.5

def analyze_business_legitimacy(business_features):
    """Analyze business legitimacy indicators"""
    try:
        # Low business legitimacy scores are high risk for wire transfers
        legitimacy_score = np.mean(business_features)
        
        # Invert the score - low legitimacy = high risk
        risk_score = 1.0 - legitimacy_score
        
        # Apply wire transfer specific adjustments
        if legitimacy_score < 0.3:  # Very low legitimacy
            risk_score = min(1.0, risk_score + 0.2)
            
        return risk_score
        
    except Exception as e:
        logger.error(f"Business legitimacy analysis error: {e}")
        return 0.5
