from datetime import time
from typing import List, Dict
import pandas as pd

from yubaoCtrader.app.portfolio_strategy import StrategyEngine, StrategyTemplate
from yubaoCtrader.trader.constant import Direction
from yubaoCtrader.trader.object import TickData, BarData, TradeData
from yubaoCtrader.trader.utility import DailyBarGenerator, ArrayManager, MinuteBarsGenerator, TradeResult


class TurtleStrategy50(StrategyTemplate):
    """海龟策略"""

    author: str = "yubao"

    # 参数
    entry_window: int = 20
    exit_window: int = 10
    cci_window: int = 14
    cci_signal: int = 20
    n_window: int = 20
    unit_limit: int = 4
    price_add: int = 10
    capital: int = 20_000_000
    risk_level: float = 0.002


    # 名称列表
    parameters = [
        "entry_window",
        "exit_window",
        "cci_window",
        "cci_signal",
        "n_window",
        "unit_limit",
        "price_add",
        "capital",
        "risk_level"
    ]
    variables = []


    def __init__(
        self, 
        strategy_engine: "StrategyEngine", 
        strategy_name: str,
        vt_symbols: List[str],
        setting: dict
    ) -> None:
        """构造函数"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)

        # 加载指数合约信息
        df: pd.DataFrame = pd.read_csv("index_contract.csv")
        self.contract_sizes: Dict[str, int] = {row.vt_symbol: row.contract_size for _, row in df.iterrows()}
        for _, row in df.iterrows():
            key = get_product_name(row.vt_symbol)
            self.contract_sizes[key] = row.contract_size

        # 初始化信号字典
        self.signals: Dict[str, TurtleSignal] = {}
        for vt_symbol in vt_symbols:
            # 读取合约对应的乘数
            key = get_product_name(vt_symbol)
            contract_size = self.contract_sizes[vt_symbol]

            self.signals[vt_symbol] = TurtleSignal(
                vt_symbol,
                self.entry_window,
                self.exit_window,
                self.cci_window,
                self.cci_signal,
                self.n_window,
                self.unit_limit,
                contract_size,
                self.capital,
                self.risk_level
            )

        # 初始化目标字典
        self.targets: Dict[str, int] = {}
        
        # 初始化分钟K线合成器
        self.mbg: MinuteBarsGenerator = MinuteBarsGenerator(self.on_bars)

    def on_init(self):
        """初始化"""
        self.write_log("策略初始化")
        self.load_bars(20)
    
    def on_start(self):
        """启动"""
        self.write_log("策略启动")

    def on_stop(self):
        """停止"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """Tick推送"""
        self.mbg.update_tick(tick)

    def on_bars(self, bars: Dict[str, BarData]):
        """原始k线推送"""
        bar = list(bars.values())[0]
        self.write_log(f"{bar.datetime} - {self.pos_data}")
        # 全撤之前的委托
        self.cancel_all()

        # 计算合约目标
        self.calculate_targets(bars)

        # 发送交易委托
        self.send_orders(bars)

    def calculate_targets(self, bars: Dict[str, BarData]) -> None:
        """计算每个合约的目标"""
        for vt_symbol, bar in bars.items():
            signal: TurtleSignal = self.signals[vt_symbol]
            signal.on_bar(bar)
            self.targets[vt_symbol] = signal.get_target()

    def send_orders(self, bars: Dict[str, BarData]) -> None:
        """发送委托"""
        for vt_symbol, bar in bars.items():
            # 计算目标和实际仓位差
            target = self.targets[vt_symbol]
            pos = self.get_pos(vt_symbol)
            diff: int = target - pos

            # 基于仓位差执行交易
            if diff > 0:
                price: float = bar.close_price + self.price_add

                # 由于海龟所有开平仓都会先回到仓位0的情况
                # 因此只需要考虑本次是开仓还是平仓即可
                if pos < 0:
                    self.cover(vt_symbol, price, abs(diff))
                else:
                    self.buy(vt_symbol, price, abs(diff))
            elif diff < 0:
                price: float = bar.close_price - self.price_add

                if pos > 0:
                    self.sell(vt_symbol, price, abs(diff))
                else:
                    self.short(vt_symbol, price, abs(diff))


class TurtleSignal:
    """海龟信号"""

    def __init__(
        self,
        vt_symbol: str,
        entry_window: int,
        exit_window: int,
        cci_window: int,
        cci_signal: int,
        n_window: int,
        unit_limit: int,
        contract_size: int,
        capital: int,
        risk_level: float
    ) -> None:
        """构造函数"""
        # 参数
        self.vt_symbol: str = vt_symbol
        self.entry_window: int = entry_window
        self.exit_window: int = exit_window
        self.cci_window: int = cci_window
        self.cci_signal: int = cci_signal
        self.n_window: int = n_window
        self.unit_limit: int = unit_limit
        self.contract_size: int = contract_size
        self.capital: int = capital
        self.risk_level: float = risk_level

        # 变量
        self.target: int = 0
        self.unit: int = 0

        # 因子
        self.factor = TurtleFactor(
            entry_window=self.entry_window,
            exit_window=self.exit_window,
            cci_window=self.cci_window,
            cci_signal=self.cci_signal,
            n_window=self.n_window,
            contract_size=self.contract_size,
            unit_limit=self.unit_limit,
            pnl_filter=False
        )

    def on_bar(self, bar: BarData) -> None:
        """K线推送"""
        # 推送给因子计算
        self.factor.on_bar(bar)

        # 无仓位时更新unit
        if not self.target:
            if self.factor.n:
                self.unit = (self.capital * self.risk_level) / (self.factor.n * self.contract_size)
            self.unit = max(self.unit, 1)

        # 获取因子目标仓位
        self.target = self.factor.get_target() * self.unit
        self.target = int(self.target)

    def get_target(self) -> None:
        """获取信号"""
        return self.target


class TurtleFactor:
    """海龟因子"""

    def __init__(
        self,
        entry_window: int,
        exit_window: int,
        cci_window: int,
        cci_signal: int,
        n_window: int,
        contract_size: int,
        unit_limit: int,
        pnl_filter: bool
    ) -> None:
        """构造函数"""
        # 参数
        self.entry_window: int = entry_window
        self.exit_window: int = exit_window
        self.cci_window: int = cci_window
        self.cci_signal: int = cci_signal
        self.n_window: int = n_window
        self.contract_size: int = contract_size
        self.unit_limit: int = unit_limit
        self.pnl_filter: bool = pnl_filter

        # 变量
        self.entry_up: float = 0.0 # 入场通道上轨
        self.entry_down: float = 0.0 # 入场通道下轨

        self.exit_up: float = 0.0 # 出场通道上轨，做空时涨破上轨出场
        self.exit_down: float = 0.0 # 出场通道下轨，做多时跌破下轨出场

        self.cci: float = 0.0 # cci数值

        self.n: float = 0.0 # 波动度量

        self.long_entry: float = 0.0 # 做多开仓的价格
        self.short_entry: float = 0.0 # 做空开仓的价格

        self.target: int = 0 # 目标仓位
        self.traded: bool = False # 日内是否交易过

        # 工具
        self.am = ArrayManager()
        self.bg = DailyBarGenerator(self.on_daily_bar, time(14, 59))

    def on_bar(self, bar: BarData):
        """原始k线推送"""
        # am初始化完毕且今天没交易过，才能交易
        if self.am.inited and not self.traded:
            old_target = self.target # 在交易之前记录一下target

            # 判断当前目标
            if not self.target:
                self.check_long_target(bar)
                self.check_short_target(bar)
            elif self.target > 0:
                self.check_long_target(bar)

                long_stop: float = self.long_entry - 2 * self.n
                long_stop = max(long_stop, self.exit_down)

                if bar.low_price <= long_stop:
                    self.target = 0

            elif self.target < 0:
                self.check_short_target(bar)

                short_stop: float = self.short_entry + 2 * self.n
                short_stop = min(short_stop, self.exit_up)

                if bar.high_price >= short_stop:
                    self.target = 0

            # 如果old_target和刚刚计算出来的target不一样，说明现在要做交易
            # 今天的一次交易已经用完
            if old_target != self.target:
                self.traded = True

        self.bg.update_bar(bar)

    def on_daily_bar(self, bar: BarData):
        """日K线推送"""
        # 缓存K线序列
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 只有无持仓时，才更新入场通道位置和波动度量
        if not self.target:
            self.entry_up, self.entry_down = self.am.donchian(self.entry_window)
            self.n = self.am.atr(self.n_window)

        self.exit_up, self.exit_down = self.am.donchian(self.exit_window)

        self.cci = self.am.cci(self.cci_window)

        # 新的一天清空交易记录
        self.traded = False

    def check_long_target(self, bar: BarData):
        """设置本根K线的多头目标仓位"""
        level: int = self.unit_limit

        while level > 0:
            level_limit: int = level
            level_price = self.entry_up + self.n * (level - 1) * 0.5

            if self.target < level_limit and bar.high_price >= level_price and self.cci > self.cci_signal:
                self.target = level_limit
                self.long_entry = bar.close_price
                break

            level -= 1

    def check_short_target(self, bar: BarData):
        """设置本根K线的空头目标仓位"""
        level: int = self.unit_limit

        while level > 0:
            level_limit: int = -level
            level_price = self.entry_down - self.n * (level - 1) * 0.5

            if self.target > level_limit and bar.low_price <= level_price and self.cci < -self.cci_signal:
                self.target = level_limit
                self.short_entry = bar.close_price
                break

            level -= 1

    def get_target(self) -> int:
        """获取因子目标"""
        # 如果不进行过滤，则直接返回目标
        if not self.pnl_filter:
            return self.target

def get_product_name(vt_symbol: str) -> str:
    """获取合约代码，去除数字部分"""
    return "".join([w for w in vt_symbol if not w.isdigit()])
