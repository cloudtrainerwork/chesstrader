"""
Enhanced Strategy Recommender for actionable options trading.

Provides specific contract recommendations with entry/exit dates,
strike prices, expirations, and concrete trading instructions.
"""

import glob
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import yfinance as yf

from ..strategies.factory import StrategyFactory

logger = logging.getLogger(__name__)


class OptionsContract:
    """Represents a specific options contract"""
    def __init__(self, symbol: str, strike: float, expiration: str,
                 option_type: str, current_price: float, bid: float, ask: float):
        self.symbol = symbol
        self.strike = strike
        self.expiration = expiration
        self.option_type = option_type  # 'call' or 'put'
        self.current_price = current_price
        self.bid = bid
        self.ask = ask

    @property
    def contract_symbol(self) -> str:
        """Generate standard options contract symbol"""
        exp_date = datetime.strptime(self.expiration, '%Y-%m-%d').strftime('%y%m%d')
        option_code = 'C' if self.option_type.lower() == 'call' else 'P'
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.symbol}{exp_date}{option_code}{strike_str}"


class EnhancedStrategyRecommender:
    """
    Enhanced options strategy recommender with actionable trading information.

    Provides specific contracts, entry/exit dates, and detailed trade instructions
    rather than vague strategy names.
    """

    def __init__(self):
        self.risk_free_rate = 0.045  # Current ~4.5% risk-free rate

    def _get_option_price(self, option_row, price_type='mid'):
        """
        Get option price with fallbacks for when market is closed.

        Args:
            option_row: Pandas Series with option data
            price_type: 'bid', 'ask', 'mid', or 'last'

        Returns:
            float: Option price
        """
        bid = option_row.get('bid', 0)
        ask = option_row.get('ask', 0)
        last = option_row.get('lastPrice', 0)

        # If bid/ask are both 0 (market closed), use last price
        if bid == 0 and ask == 0:
            return max(last, 0.01)  # Minimum 1 cent

        # If only one is 0, use the other
        if bid == 0:
            return max(ask, 0.01)
        if ask == 0:
            return max(bid, 0.01)

        # Normal market hours - use requested price type
        if price_type == 'bid':
            return bid
        elif price_type == 'ask':
            return ask
        elif price_type == 'mid':
            return (bid + ask) / 2
        elif price_type == 'last':
            return last
        else:
            return (bid + ask) / 2

    # Map StrategyType enum values to the contract-builder methods we have implemented.
    # Types not in this map are skipped — we have no contract-construction logic for them.
    _BUILDER_MAP = None  # populated lazily after StrategyType is importable

    def _get_builder_map(self) -> Dict:
        if self._BUILDER_MAP is None:
            from ..strategies.base import StrategyType
            EnhancedStrategyRecommender._BUILDER_MAP = {
                StrategyType.BULL_CALL_SPREAD: self._create_bull_call_spread,
                StrategyType.BEAR_PUT_SPREAD:  self._create_bear_put_spread,
                StrategyType.IRON_CONDOR:      self._create_iron_condor,
                # ponytail: covered call ≈ short call against owned shares; no separate StrategyType
                StrategyType.SHORT_CALL:       self._create_covered_call,
                # Butterfly uses the same iron-condor builder as a reasonable approximation
                StrategyType.BUTTERFLY:        self._create_iron_condor,
            }
        return self._BUILDER_MAP

    def _find_checkpoint(self) -> Optional[str]:
        """Return the most-recently-modified RegimeDetector checkpoint, or None."""
        patterns = ['checkpoints/regime_detector*.pt', 'checkpoints/regime_detector*.pth']
        matches = [f for p in patterns for f in glob.glob(p)]
        return max(matches, key=os.path.getmtime) if matches else None

    def _heuristic_regime(self, hist: pd.DataFrame) -> int:
        """Map SMA-crossover sentiment to a RegimeType int. Fallback when no ML checkpoint."""
        from ..data.regime_labeler import RegimeType
        sentiment = self._analyze_market_sentiment(hist)
        return {
            'bullish': int(RegimeType.BULL_TRENDING),
            'bearish': int(RegimeType.BEAR_TRENDING),
        }.get(sentiment, int(RegimeType.SIDEWAYS_RANGING))

    def _detect_regime(self, symbol: str, hist: pd.DataFrame) -> int:
        """
        Classify market regime via the ML pipeline.

        Loads the most recent RegimeDetector checkpoint if one exists, runs the
        48-dimensional feature vector through it, and returns the predicted regime
        index.  Falls back to a simple SMA-crossover heuristic when no checkpoint
        is present or the pipeline raises.

        Note: RegimeStateVector.calculate() fetches its own price data internally,
        so hist is fetched twice in the ML path.  ponytail: consolidate when
        RegimeStateVector accepts pre-fetched data.
        """
        checkpoint = self._find_checkpoint()
        if not checkpoint:
            logger.info("No RegimeDetector checkpoint found; using heuristic regime detection")
            return self._heuristic_regime(hist)

        try:
            import torch
            from ..models.regime_detector import RegimeDetector
            from ..features.regime_features import RegimeStateVector

            model = RegimeDetector()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            model.eval()

            state_vector = RegimeStateVector().calculate(symbol)
            x = torch.tensor(state_vector.values, dtype=torch.float32).unsqueeze(0)
            predicted, _, _ = model.predict_regime(x)
            regime = int(predicted[0].item())
            logger.info(f"ML regime detection: {regime} for {symbol}")
            return regime
        except Exception as exc:
            logger.warning(f"ML regime detection failed ({exc}); using heuristic")
            return self._heuristic_regime(hist)

    def get_actionable_recommendations(self, symbol: str, analysis_days: int = 5) -> List[Dict[str, Any]]:
        """
        Get actionable options recommendations with specific contracts and dates.

        Uses the ML pipeline (RegimeDetector → StrategyFactory) for regime-aware
        strategy selection when a trained checkpoint is available; falls back to
        a heuristic SMA-crossover regime when not.

        Args:
            symbol: Stock symbol to analyze
            analysis_days: Number of days to look ahead for opportunities

        Returns:
            List of actionable trading recommendations with specific contracts
        """
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="60d", interval="1d")
            current_price = float(hist['Close'].iloc[-1])

            expirations = stock.options
            if not expirations:
                return self._fallback_recommendations(symbol, current_price)

            volatility = self._calculate_implied_volatility(hist)

            # Regime detection: try ML pipeline, fall back to heuristic
            regime = self._detect_regime(symbol, hist)

            # Strategy selection via StrategyFactory
            factory = StrategyFactory()
            factory_recs = factory.get_recommended_strategies(regime, max_recommendations=5)

            # Choose expiration (25-50 DTE; fall back to nearest available)
            target_expiration = next(
                (e for e in expirations
                 if 25 <= (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days <= 50),
                expirations[0]
            )

            try:
                option_chain = stock.option_chain(target_expiration)
            except Exception as e:
                logger.warning(f"Could not fetch option chain: {e}")
                return self._fallback_recommendations(symbol, current_price)

            builder_map = self._get_builder_map()
            recommendations = []
            for factory_rec in factory_recs:
                builder = builder_map.get(factory_rec.strategy_type)
                if builder is None:
                    continue
                rec = builder(symbol, current_price, option_chain, target_expiration, volatility)
                if rec:
                    recommendations.append(rec)

            recommendations.sort(key=lambda x: x['profit_risk_ratio'], reverse=True)
            return recommendations[:3]

        except Exception as e:
            logger.error(f"Error generating enhanced recommendations for {symbol}: {e}")
            return self._fallback_recommendations(symbol, 100.0)

    def _analyze_market_sentiment(self, hist: pd.DataFrame) -> str:
        """Analyze recent price action to determine market sentiment"""
        recent_returns = hist['Close'].pct_change().tail(10)
        avg_return = recent_returns.mean()

        # Check for trend
        sma_5 = hist['Close'].rolling(5).mean().iloc[-1]
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]

        if avg_return > 0.01 and sma_5 > sma_20:
            return "bullish"
        elif avg_return < -0.01 and sma_5 < sma_20:
            return "bearish"
        else:
            return "neutral"

    def _calculate_implied_volatility(self, hist: pd.DataFrame) -> float:
        """Calculate historical volatility as proxy for IV"""
        returns = hist['Close'].pct_change().dropna()
        return float(returns.std() * np.sqrt(252))  # Annualized volatility

    def _create_bull_call_spread(self, symbol: str, current_price: float,
                               option_chain, expiration: str, volatility: float) -> Dict[str, Any]:
        """Create a bull call spread recommendation with specific contracts"""
        calls = option_chain.calls

        # Find ATM and OTM strikes
        long_strike = self._find_closest_strike(calls, current_price)
        short_strike = self._find_closest_strike(calls, current_price * 1.05)  # 5% OTM

        long_call = calls[calls['strike'] == long_strike].iloc[0] if len(calls[calls['strike'] == long_strike]) > 0 else None
        short_call = calls[calls['strike'] == short_strike].iloc[0] if len(calls[calls['strike'] == short_strike]) > 0 else None

        if long_call is None or short_call is None:
            return None

        # Get option prices with fallbacks
        long_price = self._get_option_price(long_call, 'ask')
        short_price = self._get_option_price(short_call, 'bid')

        logger.info(f"Bull call spread: Long ${long_strike} @ ${long_price:.2f}, Short ${short_strike} @ ${short_price:.2f}")

        # Calculate trade metrics
        net_debit = long_price - short_price
        max_profit = (short_strike - long_strike) - net_debit
        max_loss = net_debit
        breakeven = long_strike + net_debit

        # Calculate confidence based on probability of profit
        prob_profit = self._calculate_probability_above(current_price, breakeven, volatility,
                                                      self._days_to_expiration(expiration))
        confidence = min(prob_profit * 100, 85)  # Cap at 85%

        return {
            'strategy_name': 'Bull Call Spread',
            'strategy_type': 'Vertical Spread',
            'market_outlook': 'Moderately Bullish',
            'confidence': round(confidence, 1),
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'exit_date': expiration,
            'contracts': [
                {
                    'action': 'BUY',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}C{int(long_strike*1000):08d}",
                    'strike': long_strike,
                    'expiration': expiration,
                    'type': 'call',
                    'price': long_price,
                    'description': f"Buy {symbol} {long_strike} Call"
                },
                {
                    'action': 'SELL',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}C{int(short_strike*1000):08d}",
                    'strike': short_strike,
                    'expiration': expiration,
                    'type': 'call',
                    'price': short_price,
                    'description': f"Sell {symbol} {short_strike} Call"
                }
            ],
            'trade_details': {
                'net_debit': round(net_debit, 2),
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'breakeven': round(breakeven, 2),
                'profit_risk_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
                'probability_of_profit': f"{prob_profit:.1%}"
            },
            'exit_strategy': {
                'target_profit': f"Close at 50% of max profit (${max_profit * 0.5:.2f})",
                'stop_loss': f"Close at 200% of premium paid (${net_debit * 2:.2f} loss)",
                'time_decay': "Consider closing with 7-10 days to expiration"
            },
            'current_price': current_price,
            'days_to_expiration': self._days_to_expiration(expiration),
            'profit_risk_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0
        }

    def _create_bear_put_spread(self, symbol: str, current_price: float,
                              option_chain, expiration: str, volatility: float) -> Dict[str, Any]:
        """Create a bear put spread recommendation"""
        puts = option_chain.puts

        # Find ATM and OTM strikes
        long_strike = self._find_closest_strike(puts, current_price)
        short_strike = self._find_closest_strike(puts, current_price * 0.95)  # 5% OTM

        long_put = puts[puts['strike'] == long_strike].iloc[0] if len(puts[puts['strike'] == long_strike]) > 0 else None
        short_put = puts[puts['strike'] == short_strike].iloc[0] if len(puts[puts['strike'] == short_strike]) > 0 else None

        if long_put is None or short_put is None:
            return None

        # Get option prices with fallbacks
        long_price = self._get_option_price(long_put, 'ask')
        short_price = self._get_option_price(short_put, 'bid')

        net_debit = long_price - short_price
        max_profit = (long_strike - short_strike) - net_debit
        max_loss = net_debit
        breakeven = long_strike - net_debit

        prob_profit = self._calculate_probability_below(current_price, breakeven, volatility,
                                                       self._days_to_expiration(expiration))
        confidence = min(prob_profit * 100, 85)

        return {
            'strategy_name': 'Bear Put Spread',
            'strategy_type': 'Vertical Spread',
            'market_outlook': 'Moderately Bearish',
            'confidence': round(confidence, 1),
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'exit_date': expiration,
            'contracts': [
                {
                    'action': 'BUY',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}P{int(long_strike*1000):08d}",
                    'strike': long_strike,
                    'expiration': expiration,
                    'type': 'put',
                    'price': long_price,
                    'description': f"Buy {symbol} {long_strike} Put"
                },
                {
                    'action': 'SELL',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}P{int(short_strike*1000):08d}",
                    'strike': short_strike,
                    'expiration': expiration,
                    'type': 'put',
                    'price': short_price,
                    'description': f"Sell {symbol} {short_strike} Put"
                }
            ],
            'trade_details': {
                'net_debit': round(net_debit, 2),
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'breakeven': round(breakeven, 2),
                'profit_risk_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
                'probability_of_profit': f"{prob_profit:.1%}"
            },
            'exit_strategy': {
                'target_profit': f"Close at 50% of max profit (${max_profit * 0.5:.2f})",
                'stop_loss': f"Close at 200% of premium paid (${net_debit * 2:.2f} loss)",
                'time_decay': "Consider closing with 7-10 days to expiration"
            },
            'current_price': current_price,
            'days_to_expiration': self._days_to_expiration(expiration),
            'profit_risk_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0
        }

    def _create_iron_condor(self, symbol: str, current_price: float,
                          option_chain, expiration: str, volatility: float) -> Dict[str, Any]:
        """Create an iron condor recommendation for neutral markets"""
        calls = option_chain.calls
        puts = option_chain.puts

        # Iron condor strikes (roughly 15-20 delta)
        short_put_strike = self._find_closest_strike(puts, current_price * 0.90)
        long_put_strike = self._find_closest_strike(puts, current_price * 0.85)
        short_call_strike = self._find_closest_strike(calls, current_price * 1.10)
        long_call_strike = self._find_closest_strike(calls, current_price * 1.15)

        # Get option prices
        try:
            short_put = puts[puts['strike'] == short_put_strike].iloc[0]
            long_put = puts[puts['strike'] == long_put_strike].iloc[0]
            short_call = calls[calls['strike'] == short_call_strike].iloc[0]
            long_call = calls[calls['strike'] == long_call_strike].iloc[0]
        except (IndexError, KeyError):
            return None

        # Get option prices with fallbacks
        short_put_price = self._get_option_price(short_put, 'bid')
        long_put_price = self._get_option_price(long_put, 'ask')
        short_call_price = self._get_option_price(short_call, 'bid')
        long_call_price = self._get_option_price(long_call, 'ask')

        net_credit = (short_put_price + short_call_price) - (long_put_price + long_call_price)
        max_profit = net_credit
        max_loss = (short_call_strike - long_call_strike) - net_credit

        # Breakevens
        upper_breakeven = short_call_strike + net_credit
        lower_breakeven = short_put_strike - net_credit

        # Probability of profit (stock stays between breakevens)
        prob_profit = self._calculate_probability_between(current_price, lower_breakeven, upper_breakeven,
                                                         volatility, self._days_to_expiration(expiration))
        confidence = min(prob_profit * 100, 75)  # Iron condors typically lower confidence

        return {
            'strategy_name': 'Iron Condor',
            'strategy_type': 'Income Strategy',
            'market_outlook': 'Neutral/Range-bound',
            'confidence': round(confidence, 1),
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'exit_date': expiration,
            'contracts': [
                {
                    'action': 'SELL',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}P{int(short_put_strike*1000):08d}",
                    'strike': short_put_strike,
                    'expiration': expiration,
                    'type': 'put',
                    'price': short_put_price,
                    'description': f"Sell {symbol} {short_put_strike} Put"
                },
                {
                    'action': 'BUY',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}P{int(long_put_strike*1000):08d}",
                    'strike': long_put_strike,
                    'expiration': expiration,
                    'type': 'put',
                    'price': long_put_price,
                    'description': f"Buy {symbol} {long_put_strike} Put"
                },
                {
                    'action': 'SELL',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}C{int(short_call_strike*1000):08d}",
                    'strike': short_call_strike,
                    'expiration': expiration,
                    'type': 'call',
                    'price': short_call_price,
                    'description': f"Sell {symbol} {short_call_strike} Call"
                },
                {
                    'action': 'BUY',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}C{int(long_call_strike*1000):08d}",
                    'strike': long_call_strike,
                    'expiration': expiration,
                    'type': 'call',
                    'price': long_call_price,
                    'description': f"Buy {symbol} {long_call_strike} Call"
                }
            ],
            'trade_details': {
                'net_credit': round(net_credit, 2),
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'upper_breakeven': round(upper_breakeven, 2),
                'lower_breakeven': round(lower_breakeven, 2),
                'profit_risk_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
                'probability_of_profit': f"{prob_profit:.1%}",
                'profitable_range': f"${lower_breakeven:.2f} - ${upper_breakeven:.2f}"
            },
            'exit_strategy': {
                'target_profit': f"Close at 25-50% of max profit (${max_profit * 0.25:.2f} - ${max_profit * 0.5:.2f})",
                'stop_loss': f"Close if trade moves against by 200% of credit (${net_credit * 2:.2f} loss)",
                'time_decay': "Consider closing with 7-14 days to expiration"
            },
            'current_price': current_price,
            'days_to_expiration': self._days_to_expiration(expiration),
            'profit_risk_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0
        }

    def _create_covered_call(self, symbol: str, current_price: float,
                           option_chain, expiration: str, volatility: float) -> Dict[str, Any]:
        """Create a covered call recommendation (assumes 100 shares owned)"""
        calls = option_chain.calls

        # OTM call (5-10% above current price)
        strike = self._find_closest_strike(calls, current_price * 1.07)
        call_option = calls[calls['strike'] == strike].iloc[0] if len(calls[calls['strike'] == strike]) > 0 else None

        if call_option is None:
            return None

        premium = self._get_option_price(call_option, 'bid')
        # premium is per share; scale it to the 100-share contract so every term
        # is in total dollars. Max profit if called away = premium + capital gain
        # to the strike; max loss if the stock goes to zero = cost basis less the
        # premium collected.
        max_profit = (premium + strike - current_price) * 100
        max_loss = (current_price - premium) * 100  # Stock could go to zero

        # Probability call expires worthless (we keep premium)
        prob_profit = 1 - self._calculate_probability_above(current_price, strike, volatility,
                                                           self._days_to_expiration(expiration))
        confidence = min(prob_profit * 100, 70)

        return {
            'strategy_name': 'Covered Call',
            'strategy_type': 'Income Strategy',
            'market_outlook': 'Neutral to Mildly Bullish',
            'confidence': round(confidence, 1),
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'exit_date': expiration,
            'prerequisites': 'Must own 100 shares of underlying stock',
            'contracts': [
                {
                    'action': 'SELL',
                    'contract_symbol': f"{symbol}{self._format_expiration(expiration)}C{int(strike*1000):08d}",
                    'strike': strike,
                    'expiration': expiration,
                    'type': 'call',
                    'price': premium,
                    'description': f"Sell {symbol} {strike} Call (covered by 100 shares)"
                }
            ],
            'trade_details': {
                'premium_collected': round(premium, 2),
                'max_profit': round(max_profit, 2),
                'downside_protection': f"${premium:.2f} per share",
                'assignment_price': strike,
                'yield_if_called': f"{((strike - current_price + premium) / current_price * 100):.1f}%",
                'probability_of_profit': f"{prob_profit:.1%}"
            },
            'exit_strategy': {
                'target_profit': "Let expire worthless to keep full premium",
                'assignment_risk': f"Stock may be called away if above ${strike:.2f} at expiration",
                'buyback_option': "Buy back call if it drops to 10-20% of premium collected"
            },
            'current_price': current_price,
            'days_to_expiration': self._days_to_expiration(expiration),
            # Reward-to-risk ratio, consistent with the spread strategies above.
            # (Previously max_profit / 1000, a fabricated constant that ranked by
            # raw dollar profit rather than risk-adjusted return.)
            'profit_risk_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0
        }

    # Helper methods
    def _find_closest_strike(self, options_df: pd.DataFrame, target_price: float) -> float:
        """Find the strike price closest to target price"""
        if options_df.empty:
            return target_price
        strikes = options_df['strike'].values
        closest_idx = np.argmin(np.abs(strikes - target_price))
        return strikes[closest_idx]

    def _format_expiration(self, expiration: str) -> str:
        """Format expiration date for options symbol (YYMMDD)"""
        return datetime.strptime(expiration, '%Y-%m-%d').strftime('%y%m%d')

    def _days_to_expiration(self, expiration: str) -> int:
        """Calculate days until expiration"""
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        return (exp_date - datetime.now()).days

    def _calculate_probability_above(self, current_price: float, target_price: float,
                                   volatility: float, days: int) -> float:
        """Calculate probability stock will be above target using Black-Scholes"""
        if days <= 0:
            return 1.0 if current_price > target_price else 0.0

        t = days / 365.0
        d1 = (np.log(current_price / target_price) + (self.risk_free_rate + 0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
        # Risk-neutral probability that S_T > target is N(d2), not N(d1).
        # N(d1) is the call delta; using it overstates the probability.
        d2 = d1 - volatility * np.sqrt(t)

        from scipy.stats import norm
        return norm.cdf(d2)

    def _calculate_probability_below(self, current_price: float, target_price: float,
                                   volatility: float, days: int) -> float:
        """Calculate probability stock will be below target"""
        return 1.0 - self._calculate_probability_above(current_price, target_price, volatility, days)

    def _calculate_probability_between(self, current_price: float, lower: float, upper: float,
                                     volatility: float, days: int) -> float:
        """Calculate probability stock will be between two prices"""
        prob_above_lower = self._calculate_probability_above(current_price, lower, volatility, days)
        prob_above_upper = self._calculate_probability_above(current_price, upper, volatility, days)
        return prob_above_lower - prob_above_upper

    def _fallback_recommendations(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Provide basic recommendations when options data is unavailable"""
        return [
            {
                'strategy_name': 'Long Call',
                'strategy_type': 'Directional',
                'market_outlook': 'Bullish',
                'confidence': 45.0,
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
                'exit_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'note': 'Options chain data unavailable - general recommendation',
                'contracts': [
                    {
                        'action': 'BUY',
                        'description': f"Buy {symbol} ATM Call (~${current_price:.2f} strike)",
                        'recommendation': 'Select call option 30-45 days to expiration'
                    }
                ],
                'trade_details': {
                    'general_advice': 'Look for options with high liquidity (tight bid-ask spread)',
                    'risk_management': 'Risk no more than 2-3% of portfolio on single trade'
                },
                'current_price': current_price,
                'profit_risk_ratio': 2.0
            }
        ]