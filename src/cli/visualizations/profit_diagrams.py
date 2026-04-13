"""
ASCII art profit/loss diagrams for options strategies.

Creates visual representations of P&L curves to help traders understand
the risk/reward profile of each strategy and spot calculation errors.
"""

from typing import Dict, List, Optional


class ProfitDiagramGenerator:
    """Generates ASCII art profit/loss diagrams for options strategies."""

    def __init__(self, width: int = 60, height: int = 15):
        """
        Initialize the diagram generator.

        Args:
            width: Width of the diagram in characters
            height: Height of the diagram in characters
        """
        self.width = width
        self.height = height

    def create_bull_call_spread(self, current_price: float, long_strike: float,
                                short_strike: float, net_debit: float,
                                max_profit: float, breakeven: float) -> List[str]:
        """
        Create a bull call spread P&L diagram.

        Pattern should show:
        - Limited loss below long strike
        - Diagonal profit zone between strikes
        - Capped profit above short strike
        """
        lines = []

        # Title
        lines.append("📈 Bull Call Spread P/L Diagram")
        lines.append("")

        # Create the chart
        chart = []

        # Top axis
        chart.append(f"Profit ↑ Max: ${max_profit:.0f}")

        # Price range for x-axis (±15% from current)
        min_price = current_price * 0.85
        max_price = current_price * 1.15

        # Build the profit curve
        for i in range(self.height - 4, -1, -1):
            line = "      "

            # Draw the curve
            for x in range(self.width - 10):
                price = min_price + (max_price - min_price) * (x / (self.width - 10))

                # Calculate P/L at this price
                if price <= long_strike:
                    # Below long strike - max loss
                    pl = -net_debit
                elif price >= short_strike:
                    # Above short strike - max profit
                    pl = max_profit
                else:
                    # Between strikes - proportional
                    pl = (price - long_strike) - net_debit

                # Scale P/L to chart height
                pl_scaled = (pl + net_debit) / (max_profit + net_debit)
                chart_level = int(pl_scaled * (self.height - 4))

                # Draw the point
                if i == chart_level:
                    if price < breakeven:
                        line += "─"
                    elif abs(price - breakeven) < (max_price - min_price) / self.width:
                        line += "●"  # Breakeven point
                    else:
                        line += "═"  # Profit zone
                elif i == 0 and pl < 0:
                    line += "_"  # Loss floor
                elif i < chart_level and price > long_strike and price < short_strike:
                    if i == chart_level - 1:
                        line += "/"  # Rising slope
                    else:
                        line += " "
                else:
                    line += " "

            chart.append(line)

        # Zero line
        zero_line = "    $0"
        for x in range(self.width - 10):
            price = min_price + (max_price - min_price) * (x / (self.width - 10))
            if price <= breakeven:
                zero_line += "─"
            else:
                zero_line += "─"
        chart.append(zero_line)

        # Loss indicator
        chart.append(f"  Loss ↓ Max: -${net_debit:.0f}")
        chart.append("")

        # X-axis with strike markers
        x_axis = "      "
        for x in range(self.width - 10):
            price = min_price + (max_price - min_price) * (x / (self.width - 10))
            if abs(price - long_strike) < (max_price - min_price) / (self.width * 2):
                x_axis += "│"
            elif abs(price - short_strike) < (max_price - min_price) / (self.width * 2):
                x_axis += "│"
            elif abs(price - current_price) < (max_price - min_price) / (self.width * 2):
                x_axis += "▲"  # Current price marker
            else:
                x_axis += " "
        chart.append(x_axis)

        # Price labels
        labels = f"     ${min_price:.0f}   Long:${long_strike:.0f}   Now:${current_price:.0f}   Short:${short_strike:.0f}   ${max_price:.0f}"
        chart.append(labels)
        chart.append("                                    Stock Price →")

        lines.extend(chart)

        # Key metrics below
        lines.append("")
        lines.append(f"🎯 Breakeven: ${breakeven:.2f} | 💰 Max Profit: ${max_profit:.2f} | 💸 Max Loss: ${net_debit:.2f}")

        return lines

    def create_bear_put_spread(self, current_price: float, long_strike: float,
                               short_strike: float, net_debit: float,
                               max_profit: float, breakeven: float) -> List[str]:
        """Create a bear put spread P&L diagram."""
        lines = []

        lines.append("📉 Bear Put Spread P/L Diagram")
        lines.append("")

        chart = []
        chart.append(f"Profit ↑ Max: ${max_profit:.0f}")

        min_price = current_price * 0.85
        max_price = current_price * 1.15

        for i in range(self.height - 4, -1, -1):
            line = "      "

            for x in range(self.width - 10):
                price = min_price + (max_price - min_price) * (x / (self.width - 10))

                # Calculate P/L at this price
                if price <= short_strike:
                    # Below short strike - max profit
                    pl = max_profit
                elif price >= long_strike:
                    # Above long strike - max loss
                    pl = -net_debit
                else:
                    # Between strikes
                    pl = (long_strike - price) - net_debit

                pl_scaled = (pl + net_debit) / (max_profit + net_debit)
                chart_level = int(pl_scaled * (self.height - 4))

                if i == chart_level:
                    if price > breakeven:
                        line += "─"
                    elif abs(price - breakeven) < (max_price - min_price) / self.width:
                        line += "●"
                    else:
                        line += "═"
                elif i == 0 and pl < 0:
                    line += "_"
                elif i < chart_level and price > short_strike and price < long_strike:
                    if i == chart_level - 1:
                        line += "\\"  # Falling slope
                    else:
                        line += " "
                else:
                    line += " "

            chart.append(line)

        # Zero line and below
        chart.append("    $0" + "─" * (self.width - 10))
        chart.append(f"  Loss ↓ Max: -${net_debit:.0f}")
        chart.append("")

        # X-axis
        x_axis = "      "
        for x in range(self.width - 10):
            price = min_price + (max_price - min_price) * (x / (self.width - 10))
            if abs(price - short_strike) < (max_price - min_price) / (self.width * 2):
                x_axis += "│"
            elif abs(price - long_strike) < (max_price - min_price) / (self.width * 2):
                x_axis += "│"
            elif abs(price - current_price) < (max_price - min_price) / (self.width * 2):
                x_axis += "▲"
            else:
                x_axis += " "
        chart.append(x_axis)

        labels = f"     ${min_price:.0f}   Short:${short_strike:.0f}   Now:${current_price:.0f}   Long:${long_strike:.0f}   ${max_price:.0f}"
        chart.append(labels)
        chart.append("                                    Stock Price →")

        lines.extend(chart)
        lines.append("")
        lines.append(f"🎯 Breakeven: ${breakeven:.2f} | 💰 Max Profit: ${max_profit:.2f} | 💸 Max Loss: ${net_debit:.2f}")

        return lines

    def create_iron_condor(self, current_price: float,
                          short_put_strike: float, long_put_strike: float,
                          short_call_strike: float, long_call_strike: float,
                          net_credit: float, max_loss: float,
                          lower_breakeven: float, upper_breakeven: float) -> List[str]:
        """
        Create an iron condor P&L diagram.

        Should show the characteristic "tent" shape:
        - Max profit between the short strikes
        - Losses outside the breakevens
        - Limited loss at the wings
        """
        lines = []

        lines.append("🦅 Iron Condor P/L Diagram (The Profit Tent)")
        lines.append("")

        chart = []
        chart.append(f"Profit ↑ Max: ${net_credit:.0f}")

        # Wider range for iron condor
        min_price = long_put_strike * 0.95
        max_price = long_call_strike * 1.05

        for i in range(self.height - 4, -1, -1):
            line = "      "

            for x in range(self.width - 10):
                price = min_price + (max_price - min_price) * (x / (self.width - 10))

                # Calculate P/L at this price
                if price <= long_put_strike:
                    # Max loss on put side
                    pl = -max_loss
                elif price <= short_put_strike:
                    # Loss slope on put side
                    pl = net_credit - (short_put_strike - price)
                elif price >= long_call_strike:
                    # Max loss on call side
                    pl = -max_loss
                elif price >= short_call_strike:
                    # Loss slope on call side
                    pl = net_credit - (price - short_call_strike)
                else:
                    # Between short strikes - max profit
                    pl = net_credit

                # Scale P/L to chart height
                pl_range = net_credit + max_loss
                pl_scaled = (pl + max_loss) / pl_range
                chart_level = int(pl_scaled * (self.height - 4))

                # Draw the point
                if i == chart_level:
                    if price >= lower_breakeven and price <= upper_breakeven:
                        line += "═"  # Profit zone
                    elif abs(price - lower_breakeven) < (max_price - min_price) / self.width:
                        line += "●"  # Lower breakeven
                    elif abs(price - upper_breakeven) < (max_price - min_price) / self.width:
                        line += "●"  # Upper breakeven
                    else:
                        line += "─"  # Loss zone
                elif i == chart_level - 1:
                    # Draw the tent slopes
                    if price > long_put_strike and price < short_put_strike:
                        line += "/"  # Rising from put side
                    elif price > short_call_strike and price < long_call_strike:
                        line += "\\"  # Falling to call side
                    else:
                        line += " "
                else:
                    line += " "

            chart.append(line)

        # Zero line
        chart.append("    $0" + "─" * (self.width - 10))
        chart.append(f"  Loss ↓ Max: -${max_loss:.0f}")
        chart.append("")

        # X-axis with strike markers
        x_axis = "      "
        for x in range(self.width - 10):
            price = min_price + (max_price - min_price) * (x / (self.width - 10))
            if abs(price - current_price) < (max_price - min_price) / (self.width * 2):
                x_axis += "▲"  # Current price
            elif any(abs(price - s) < (max_price - min_price) / (self.width * 2)
                    for s in [long_put_strike, short_put_strike, short_call_strike, long_call_strike]):
                x_axis += "│"
            else:
                x_axis += " "
        chart.append(x_axis)

        # Compact labels
        labels = f"    LP:{long_put_strike:.0f} SP:{short_put_strike:.0f}  Now:${current_price:.0f}  SC:{short_call_strike:.0f} LC:{long_call_strike:.0f}"
        chart.append(labels)
        chart.append("                                    Stock Price →")

        lines.extend(chart)
        lines.append("")
        lines.append(f"🎯 Profit Zone: ${lower_breakeven:.2f} - ${upper_breakeven:.2f}")
        lines.append(f"💰 Max Profit: ${net_credit:.2f} | 💸 Max Loss: ${max_loss:.2f}")

        return lines

    def create_covered_call(self, current_price: float, strike: float,
                           premium: float, cost_basis: float) -> List[str]:
        """
        Create a covered call P&L diagram.

        Shows:
        - Linear gains up to strike
        - Capped profit above strike
        - Downside exposure (minus premium)
        """
        lines = []

        lines.append("📞 Covered Call P/L Diagram")
        lines.append("")

        chart = []

        # Calculate correct max profit (premium + potential stock gain)
        max_profit_per_share = premium + max(0, strike - cost_basis)

        chart.append(f"Profit ↑ Max: ${max_profit_per_share:.2f}/share")

        min_price = cost_basis * 0.80
        max_price = strike * 1.20

        for i in range(self.height - 4, -1, -1):
            line = "      "

            for x in range(self.width - 10):
                price = min_price + (max_price - min_price) * (x / (self.width - 10))

                # Calculate P/L at this price
                if price >= strike:
                    # Stock called away - max profit
                    pl = premium + (strike - cost_basis)
                else:
                    # Stock not called - keep premium + unrealized gain/loss
                    pl = premium + (price - cost_basis)

                # Scale P/L to chart height
                max_loss = cost_basis - premium  # If stock goes to zero
                pl_range = max_profit_per_share + max_loss
                pl_scaled = (pl + max_loss) / pl_range if pl_range > 0 else 0.5
                chart_level = int(pl_scaled * (self.height - 4))

                # Draw the point
                if i == chart_level:
                    if price < strike:
                        if price < cost_basis - premium:
                            line += "─"  # Loss zone
                        elif price < cost_basis:
                            line += "═"  # Reduced loss (premium cushion)
                        else:
                            line += "/"  # Rising profit
                    else:
                        line += "═"  # Capped profit
                elif i == 0 and pl < 0:
                    line += "_"
                else:
                    line += " "

            chart.append(line)

        # Zero line
        breakeven = cost_basis - premium
        zero_line = "    $0"
        for x in range(self.width - 10):
            price = min_price + (max_price - min_price) * (x / (self.width - 10))
            if abs(price - breakeven) < (max_price - min_price) / self.width:
                zero_line += "●"
            else:
                zero_line += "─"
        chart.append(zero_line)

        max_loss = cost_basis - premium
        chart.append(f"  Loss ↓ Max: -${max_loss:.2f}/share")
        chart.append("")

        # X-axis
        x_axis = "      "
        for x in range(self.width - 10):
            price = min_price + (max_price - min_price) * (x / (self.width - 10))
            if abs(price - current_price) < (max_price - min_price) / (self.width * 2):
                x_axis += "▲"
            elif abs(price - strike) < (max_price - min_price) / (self.width * 2):
                x_axis += "│"
            elif abs(price - cost_basis) < (max_price - min_price) / (self.width * 2):
                x_axis += "◆"  # Cost basis marker
            else:
                x_axis += " "
        chart.append(x_axis)

        labels = f"     ${min_price:.0f}    Cost:${cost_basis:.0f}   Now:${current_price:.0f}   Strike:${strike:.0f}    ${max_price:.0f}"
        chart.append(labels)
        chart.append("                                    Stock Price →")

        lines.extend(chart)
        lines.append("")
        lines.append(f"🎯 Breakeven: ${breakeven:.2f} | 💰 Max Profit: ${max_profit_per_share:.2f}/share")
        lines.append(f"📌 Strike: ${strike:.2f} | Premium Protection: ${premium:.2f}")

        return lines

    def create_straddle(self, current_price: float, strike: float,
                       call_premium: float, put_premium: float) -> List[str]:
        """
        Create a long straddle P&L diagram.

        Shows the characteristic V-shape:
        - Profit on large moves in either direction
        - Max loss at strike price
        """
        lines = []

        lines.append("🎪 Long Straddle P/L Diagram (The V-Shape)")
        lines.append("")

        total_premium = call_premium + put_premium
        upper_breakeven = strike + total_premium
        lower_breakeven = strike - total_premium

        chart = []
        chart.append("Profit ↑ Unlimited potential")

        min_price = strike * 0.70
        max_price = strike * 1.30

        for i in range(self.height - 4, -1, -1):
            line = "      "

            for x in range(self.width - 10):
                price = min_price + (max_price - min_price) * (x / (self.width - 10))

                # Calculate P/L
                if price <= strike:
                    # Put is in the money
                    pl = (strike - price) - total_premium
                else:
                    # Call is in the money
                    pl = (price - strike) - total_premium

                # Scale P/L
                max_display_profit = total_premium * 2
                pl_range = max_display_profit + total_premium
                pl_scaled = (pl + total_premium) / pl_range
                chart_level = int(pl_scaled * (self.height - 4))

                if i == chart_level:
                    if pl > 0:
                        line += "═"  # Profit zone
                    elif abs(price - lower_breakeven) < (max_price - min_price) / self.width:
                        line += "●"
                    elif abs(price - upper_breakeven) < (max_price - min_price) / self.width:
                        line += "●"
                    else:
                        line += "─"  # Loss zone
                elif i == chart_level - 1 and price != strike:
                    if price < strike:
                        line += "\\"  # Left side of V
                    else:
                        line += "/"   # Right side of V
                else:
                    line += " "

            chart.append(line)

        # Zero line
        chart.append("    $0" + "─" * (self.width - 10))
        chart.append(f"  Loss ↓ Max: -${total_premium:.2f}")

        lines.extend(chart)
        lines.append("")
        lines.append(f"🎯 Lower BE: ${lower_breakeven:.2f} | Strike: ${strike:.2f} | Upper BE: ${upper_breakeven:.2f}")
        lines.append(f"💸 Max Loss: ${total_premium:.2f} (at strike) | 💰 Profit: Unlimited both directions")

        return lines


def generate_strategy_diagram(strategy_name: str, trade_details: Dict,
                             contracts: List[Dict], current_price: float) -> List[str]:
    """
    Generate the appropriate P&L diagram based on strategy type.

    Args:
        strategy_name: Name of the strategy
        trade_details: Dictionary with trade financials
        contracts: List of contract details
        current_price: Current stock price

    Returns:
        List of strings representing the ASCII diagram
    """
    generator = ProfitDiagramGenerator(width=70, height=12)

    strategy_lower = strategy_name.lower()

    if 'bull call' in strategy_lower:
        # Extract strikes from contracts
        strikes = [c['strike'] for c in contracts]
        long_strike = min(strikes)
        short_strike = max(strikes)

        return generator.create_bull_call_spread(
            current_price=current_price,
            long_strike=long_strike,
            short_strike=short_strike,
            net_debit=trade_details.get('net_debit', 0),
            max_profit=trade_details.get('max_profit', 0),
            breakeven=trade_details.get('breakeven', long_strike)
        )

    elif 'bear put' in strategy_lower:
        strikes = [c['strike'] for c in contracts]
        long_strike = max(strikes)
        short_strike = min(strikes)

        return generator.create_bear_put_spread(
            current_price=current_price,
            long_strike=long_strike,
            short_strike=short_strike,
            net_debit=trade_details.get('net_debit', 0),
            max_profit=trade_details.get('max_profit', 0),
            breakeven=trade_details.get('breakeven', long_strike)
        )

    elif 'iron condor' in strategy_lower:
        # Extract all four strikes
        put_contracts = [c for c in contracts if c['type'].lower() == 'put']
        call_contracts = [c for c in contracts if c['type'].lower() == 'call']

        put_strikes = [c['strike'] for c in put_contracts]
        call_strikes = [c['strike'] for c in call_contracts]

        return generator.create_iron_condor(
            current_price=current_price,
            short_put_strike=max(put_strikes) if put_strikes else 0,
            long_put_strike=min(put_strikes) if put_strikes else 0,
            short_call_strike=min(call_strikes) if call_strikes else 0,
            long_call_strike=max(call_strikes) if call_strikes else 0,
            net_credit=trade_details.get('net_credit', 0),
            max_loss=abs(trade_details.get('max_loss', 0)),
            lower_breakeven=trade_details.get('lower_breakeven', 0),
            upper_breakeven=trade_details.get('upper_breakeven', 0)
        )

    elif 'covered call' in strategy_lower:
        strike = contracts[0]['strike'] if contracts else 0
        premium = contracts[0]['price'] if contracts else 0

        return generator.create_covered_call(
            current_price=current_price,
            strike=strike,
            premium=premium,
            cost_basis=current_price  # Assuming we buy at current price
        )

    elif 'straddle' in strategy_lower:
        # Find call and put premiums
        call_contracts = [c for c in contracts if c['type'].lower() == 'call']
        put_contracts = [c for c in contracts if c['type'].lower() == 'put']

        strike = contracts[0]['strike'] if contracts else current_price
        call_premium = call_contracts[0]['price'] if call_contracts else 0
        put_premium = put_contracts[0]['price'] if put_contracts else 0

        return generator.create_straddle(
            current_price=current_price,
            strike=strike,
            call_premium=call_premium,
            put_premium=put_premium
        )

    else:
        # Default message for unsupported strategies
        return [
            f"📊 P/L Diagram for {strategy_name}",
            "Visual diagram not yet implemented for this strategy type.",
            f"Current Price: ${current_price:.2f}"
        ]