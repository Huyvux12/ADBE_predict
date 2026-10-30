"""Build the Vietnamese report (Markdown + HTML), its figures and the executed notebook."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT, Config, track_dir
from .evaluation import LABELS, history_name
from .market import load_prices, load_track
from .markdown_html import to_html
from .policies import load_policy, sigmoid
from .simulate import simulate
from .notebook import execute_notebook

REPORT_NAME = 'Bao_cao_REL301m_ADBE'
NOTEBOOK = ROOT / 'ADBE_REL301m.ipynb'
MAIN_MODEL = 'cem_temporal'
# Validated categorical slots (blue, orange, aqua) and a neutral reference line.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
REFERENCE = '#8a8f98'


def money(x: float) -> str:
    return f'{x:,.2f}'


def percent(ratio: float) -> str:
    return f'{100 * ratio:.2f}%'


def results_table(summary: pd.DataFrame, period: str) -> str:
    lines = ['| Chính sách | Cuối episode (USD) | Đỉnh (USD) | Cumulative reward | Phí (USD) | Kết thúc |',
             '|---|---:|---:|---:|---:|---|']
    rows = summary[(summary.period == period) & (summary.scenario == 'sell_high')]
    lines += [f'| {LABELS[r.model]} | {money(r.final_assets)} | {money(r.peak_assets)} | '
              f'{r.cumulative_reward:.6f} | {money(r.total_fees)} | {r.end_reason} |' for r in rows.itertuples()]
    return '\n'.join(lines)


def trade_table(history: pd.DataFrame, head: int = 8, tail: int = 2) -> str:
    lines = ['| Ngày | Lệnh | Số lượng | Giá khớp | Phí | Tiền mặt sau | Cổ phiếu sau | Tổng tài sản |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    trades = history[history.quantity > 0]
    for r in pd.concat([trades.head(head), trades.tail(tail)]).drop_duplicates().itertuples():
        lines.append(f'| {r.Date} | {"Mua" if r.trade == "buy" else "Bán"} | {int(r.quantity):,} | '
                     f'{r.execution_price:.5f} | {money(r.fee)} | {money(r.cash)} | {int(r.shares):,} | '
                     f'{money(r.assets)} |')
    return '\n'.join(lines)


def replay_table(output: Path) -> str:
    s = pd.read_csv(Path(output) / 'replay/summary.csv')
    s = s[(s.model == MAIN_MODEL) & (s.scenario == 'sell_high')]
    lines = ['| Cửa sổ test 2 năm | Tài sản cuối (USD) | Đỉnh (USD) | Lợi nhuận cuối |', '|---|---:|---:|---:|']
    lines += [f'| {r.window_start} → {r.window_end} | {money(r.final_assets)} | {money(r.peak_assets)} | '
              f'{r.return_pct:+.2f}% |' for r in s.itertuples()]
    return '\n'.join(lines)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_figures(config: Config, output: Path) -> None:
    plt.switch_backend('Agg')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    folder = track_dir(output, config.main_track)
    figures = Path(output) / 'figures'
    read = lambda model, period: pd.read_csv(folder / 'histories' / history_name(model, period, 'sell_high'),
                                             parse_dates=['Date'])
    win = config.rules.win_assets / 1000
    split = config.tracks[config.main_track]
    years = lambda a, b: f'{pd.Timestamp(a).year}–{pd.Timestamp(b).year}'
    train_years, test_years = years(split.train_start, split.train_end), years(split.test_start, split.test_end)

    def win_line(ax, assets_k, **kw):
        if assets_k.max() > .6 * win:      # only when the 1M target is in view
            ax.axhline(win, color=REFERENCE, ls='--', lw=1, **kw)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, period, title in zip(axes, ['train', 'test'], [f'Train {train_years}', f'Test {test_years} (load model đã lưu)']):
        top = 0
        for model, color in zip(['penalty_aware_hold', 'cem_validation', MAIN_MODEL], SERIES):
            h = read(model, period)
            ax.plot(h.Date, h.assets / 1000, lw=2, color=color, label=LABELS[model])
            top = max(top, h.assets.max() / 1000)
        win_line(ax, pd.Series([top]), label='Mốc thắng 1 triệu USD')
        ax.set(title=title, ylabel='Tổng tài sản (nghìn USD)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=.2)
        ax.tick_params(axis='x', rotation=25)
    save_figure(fig, figures / 'equity.png')

    h = read(MAIN_MODEL, 'test')
    track = load_track(config, config.main_track)
    high = pd.Series(track.market.high, index=track.market.dates).reindex(h.Date).to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True)
    axes[0].plot(h.Date, high, color='#334155', lw=1.2, label='Giá High')
    for kind, marker, color, label in [('buy', '^', SERIES[2], 'Mua'), ('sell', 'v', SERIES[1], 'Bán')]:
        m = (h.trade == kind).to_numpy()
        axes[0].scatter(h.Date[m], high[m], marker=marker, color=color, s=40, label=label, zorder=3)
    axes[0].set_ylabel('ADBE High (USD điều chỉnh)')
    axes[0].legend(fontsize=8)
    axes[1].plot(h.Date, h.assets / 1000, color=SERIES[0], lw=2)
    win_line(axes[1], h.assets / 1000)
    axes[1].set_ylabel('Tổng tài sản (nghìn USD)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f'Test {test_years}: điểm mua/bán và tổng tài sản')
    save_figure(fig, figures / 'test_trades.png')

    log = pd.read_csv(folder / 'learning/temporal_train.csv')
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for (seed, g), color in zip(log.groupby('seed'), SERIES):
        ax.plot(g.generation, g.best_peak / 1000, lw=2, color=color, label=f'seed {seed}')
    win_line(ax, log.best_peak / 1000, label='Mốc thắng 1 triệu USD')
    ax.set(xlabel='Thế hệ CEM', ylabel='Đỉnh train tốt nhất (nghìn USD)',
           title='CEM theo thời gian: tiến trình học trên train')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=.2)
    save_figure(fig, figures / 'learning.png')


def simple_ceiling(config: Config, folder: Path) -> float:
    """Best test peak among constant-exposure policies (50%..100%) with the main model's controls."""
    schedule = load_policy(folder / f'models/{MAIN_MODEL}.json', config.rules)
    track = load_track(config, config.main_track)
    exposures = np.clip(np.arange(.50, 1.0001, .01), 1e-3, 1 - 1e-5)
    thetas = np.repeat(schedule.theta[None, :], len(exposures), axis=0)
    thetas[:, :schedule.knots] = np.log(exposures / (1 - exposures))[:, None]
    peaks, _, _ = simulate(schedule.population(thetas, track.market), *track.ranges['test'], config.rules)
    return float(peaks.max())


def report_context(config: Config, output: Path) -> dict:
    main, gap = (pd.read_csv(track_dir(output, t) / 'summary.csv') for t in [config.main_track, 'dotcom'])
    folder = track_dir(output, config.main_track)
    pick = lambda model, period, scenario='sell_high': main[(main.model == model) & (main.period == period)
                                                            & (main.scenario == scenario)].iloc[0]
    histories = {p: pd.read_csv(folder / 'histories' / history_name(MAIN_MODEL, p, 'sell_high')) for p in
                 ['train', 'test']}
    continuation = json.loads((folder / 'learning/continuation.json').read_text(encoding='utf-8'))
    learning = pd.read_csv(folder / 'learning/temporal_train.csv')
    verification = json.loads((Path(output) / 'verification.json').read_text(encoding='utf-8'))
    hold = pick('penalty_aware_hold', 'test')
    main_spec = json.loads((folder / f'models/{MAIN_MODEL}.json').read_text(encoding='utf-8'))
    main_theta, main_knots = main_spec['theta'], main_spec['knots']
    closes = load_prices(config.data_file).set_index('Date').Close

    ctx = {'continuation_logit': f"{continuation['continuation_logit']:.3f}",
           'continuation_exposure': f"{100 * continuation['continuation_exposure']:.0f}%",
           'knots_trained': continuation['schedule_knots_trained'],
           'knots_total': continuation['schedule_knots_total'],
           'schedule_last_date': continuation['schedule_last_training_date'],
           'hold_train_peak': money(pick('penalty_aware_hold', 'train').peak_assets),
           'indicator_train_peak': money(pick('cem_train_peak', 'train').peak_assets),
           'last_knot_exposure': f"{100 * sigmoid(main_theta[main_knots - 1]):.0f}%",
           'continuation_test': money(pick('cem_temporal_continuation', 'test').peak_assets),
           'simple_ceiling': money(simple_ceiling(config, folder)),
           'test_history_path': f'outputs/{config.main_track}/histories/{history_name(MAIN_MODEL, "test", "sell_high")}',
           'trade_table': trade_table(histories['test']),
           'train_table': results_table(main, 'train'), 'test_table': results_table(main, 'test'),
           'gap_train_table': results_table(gap, 'train'), 'gap_test_table': results_table(gap, 'test'),
           'replay_table': replay_table(output),
           'hold_final': money(hold.final_assets), 'hold_fees': money(hold.total_fees),
           'first_generation_peak': money(learning[learning.generation == 0].best_peak.max()),
           'audited_histories': verification['audited_histories']}
    dot = lambda model, period: gap[(gap.model == model) & (gap.period == period) & (gap.scenario == 'sell_high')].iloc[0]
    for period in ['train', 'test']:
        r = dot(MAIN_MODEL, period)
        ctx.update({f'dotcom_{period}_peak': money(r.peak_assets), f'dotcom_{period}_ratio': percent(r.score_ratio_peak),
                    f'dotcom_{period}_reason': r.end_reason})
    ctx['dotcom_hold_test'] = money(dot('penalty_aware_hold', 'test').peak_assets)
    day = lambda d: pd.Timestamp(d).strftime('%d/%m/%Y')
    for name, split in [('main', config.tracks[config.main_track]), ('dotcom', config.tracks['dotcom'])]:
        ctx.update({f'{name}_train_span': f'[{day(split.train_start)}, {day(split.train_end)})',
                    f'{name}_fit_span': f'[{day(split.train_start)}, {day(split.validation_start)})',
                    f'{name}_validation_span': f'[{day(split.validation_start)}, {day(split.train_end)})',
                    f'{name}_test_span': f'[{day(split.test_start)}, {day(split.test_end)})'})
    for period, split in load_track(config, config.main_track).describe().items():
        if isinstance(split, dict):
            ctx.update({f'{period}_first': split['first_session'], f'{period}_last': split['last_session'],
                        f'{period}_sessions': split['sessions']})
    for period in ['train', 'test']:
        r, close, last = pick(MAIN_MODEL, period), pick(MAIN_MODEL, period, 'sell_close'), histories[period].iloc[-1]
        trades = histories[period].trade
        ctx.update({f'{period}_{key}': value for key, value in {
            'peak': money(r.peak_assets), 'final': money(r.final_assets), 'ratio': percent(r.score_ratio_peak),
            'peak_date': r.peak_date, 'reward': f'{r.cumulative_reward:.6f}', 'trades': int(r.trades),
            'fees': money(r.total_fees), 'penalty': money(r.total_penalty), 'drawdown': percent(r.max_drawdown),
            'stop_sessions': int(r.sessions), 'end_reason': r.end_reason, 'close_peak': money(close.peak_assets),
            'cash': money(last.cash), 'shares': f'{int(last.shares):,}', 'mark': f'{last.mark_price:.6f}',
            'assets': money(last.assets),
            'close_valued': money(last.cash + last.shares * closes[pd.Timestamp(last.Date)]),
            'buys': int((trades == 'buy').sum()),
            'sells': int((trades == 'sell').sum()),
            'end_label': (f'Thắng ngày {r.peak_date}' if r.end_reason == 'win'
                          else f'Hết dữ liệu (đỉnh ngày {r.peak_date})')}.items()})
    return ctx


def build_report(config: Config, output: Path) -> None:
    draw_figures(config, output)
    template = (Path(__file__).with_name('report_template.md')).read_text(encoding='utf-8')
    markdown = template.format(**report_context(config, output))
    (ROOT / f'{REPORT_NAME}.md').write_text(markdown, encoding='utf-8')
    (ROOT / f'{REPORT_NAME}.html').write_text(to_html(markdown, ROOT, 'Báo cáo REL301m ADBE'), encoding='utf-8')
    execute_notebook(NOTEBOOK)
    print(f'Report: {REPORT_NAME}.html, notebook: {NOTEBOOK.name}', flush=True)
