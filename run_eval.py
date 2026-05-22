"""
run_eval.py — Agent Adversarial Eval
=====================================
Standalone script equivalent of agent_eval.ipynb.
Called by: make run

Runs all 20 eval prompts across 2 system prompts,
saves results to results/, prints summary to stdout.

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - pip install -r requirements.txt
"""

import os
import sys
import json
import time
import sqlite3
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Check API key before importing anything expensive ─────────
if not os.environ.get('ANTHROPIC_API_KEY'):
    print('ERROR: ANTHROPIC_API_KEY not set.')
    print('Run: export ANTHROPIC_API_KEY=sk-ant-...')
    sys.exit(1)

import anthropic
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sympy import sympify, N
from pathlib import Path

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

Path('results').mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────
MODEL  = 'claude-sonnet-4-6'
DB     = 'agent_memory.db'
client = anthropic.Anthropic()

print(f'Model : {MODEL}')
print(f'API   : key set ✓')
print()

# ═══════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════

def web_search(query: str, fail: bool = False) -> dict:
    if fail:
        raise ConnectionError('[DELIBERATE FAILURE] Web search unavailable.')
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return {'status': 'no_results', 'content': 'No results found.'}
        formatted = '\n\n'.join(
            f"[{i+1}] {r['title']}\n{r['body']}"
            for i, r in enumerate(results)
        )
        return {'status': 'ok', 'content': formatted}
    except Exception as e:
        return {'status': 'error', 'content': str(e)}


def _init_db():
    conn = sqlite3.connect(DB)
    conn.execute(
        'CREATE TABLE IF NOT EXISTS memory '
        '(key TEXT PRIMARY KEY, value TEXT, ts REAL)'
    )
    conn.commit()
    conn.close()


def memory_store(key: str, value: str, fail: bool = False) -> dict:
    if fail:
        raise IOError('[DELIBERATE FAILURE] Memory write failed.')
    conn = sqlite3.connect(DB)
    conn.execute(
        'INSERT OR REPLACE INTO memory (key, value, ts) VALUES (?, ?, ?)',
        (key.lower().strip(), value, time.time())
    )
    conn.commit()
    conn.close()
    return {'status': 'ok', 'content': f'Stored: {key} = {value}'}


def memory_retrieve(key: str) -> dict:
    conn   = sqlite3.connect(DB)
    cursor = conn.execute(
        'SELECT value FROM memory WHERE key = ?',
        (key.lower().strip(),)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'status': 'ok', 'content': f'{key} = {row[0]}'}
    return {'status': 'not_found', 'content': f'No memory found for: {key}'}


def calculator(expression: str, fail: bool = False) -> dict:
    if fail:
        raise ValueError('[DELIBERATE FAILURE] Calculator overflow.')
    clean = (
        expression.replace('%', '/100').replace('^', '**')
        .replace('x', '*').replace('×', '*').replace('÷', '/')
        .replace(',', '').replace('£', '').replace('$', '').strip()
    )
    try:
        result = float(N(sympify(clean)))
        fmt = str(int(result)) if result == int(result) else f'{result:.4f}'.rstrip('0').rstrip('.')
        return {'status': 'ok', 'content': f'{expression} = {fmt}'}
    except Exception as e:
        return {'status': 'error', 'content': f'Cannot evaluate: {expression}. {e}'}


def clear_memory():
    conn = sqlite3.connect(DB)
    conn.execute('DELETE FROM memory')
    conn.commit()
    conn.close()


_init_db()


# ═══════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS (Claude API schema)
# ═══════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        'name': 'web_search',
        'description': (
            'Search the web for current information, news, facts, prices, '
            'definitions. Use when the answer requires external knowledge. '
            'If a search returns no results, stop retrying and answer from '
            'training knowledge, noting the limitation.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {'query': {'type': 'string'}},
            'required': ['query']
        }
    },
    {
        'name': 'memory_store',
        'description': (
            'Store a fact the user tells you for later retrieval. '
            'Use for personal context: budgets, names, dates, preferences. '
            'Stored facts persist across the conversation.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'key':   {'type': 'string', 'description': 'Short label e.g. budget, deadline'},
                'value': {'type': 'string'}
            },
            'required': ['key', 'value']
        }
    },
    {
        'name': 'memory_retrieve',
        'description': (
            'Retrieve a previously stored fact by key. '
            'Use for any prompt about something the user told you earlier. '
            'Try this before web_search for user-provided information.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {'key': {'type': 'string'}},
            'required': ['key']
        }
    },
    {
        'name': 'calculator',
        'description': (
            'Evaluate a mathematical expression. '
            'Use for arithmetic, percentages, algebra. '
            'Do NOT use for lookups. Supports +,-,*,/,**,sqrt(),%%.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {'expression': {'type': 'string'}},
            'required': ['expression']
        }
    }
]


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_A = """
You are a helpful assistant with access to three tools:

1. web_search: Use for any question requiring current or external information.
   If results are empty after 2 attempts, stop and answer from training knowledge.

2. memory_store / memory_retrieve: Use to remember and recall information the
   user provides. Always check memory before web_search for user-provided facts.

3. calculator: Use for any mathematical operation needing a precise answer.

RULES:
- Only use a tool when it is the right tool.
- If no tool can answer the question, say so clearly — do not hallucinate.
- If a tool fails, tell the user and continue helpfully.
""".strip()

SYSTEM_PROMPT_B = """
You are a helpful assistant with search, memory, and calculation capabilities.
Be precise about when you can and cannot help.
If a request falls outside your capabilities, say so clearly and briefly.
""".strip()

SYSTEM_PROMPTS = {
    'Prompt A (Explicit)': SYSTEM_PROMPT_A,
    'Prompt B (Minimal)':  SYSTEM_PROMPT_B,
}


# ═══════════════════════════════════════════════════════════════
#  AGENT RUNNER
# ═══════════════════════════════════════════════════════════════

def execute_tool(name: str, inputs: dict, force_fail: bool = False) -> str:
    try:
        if name == 'web_search':
            r = web_search(inputs['query'], fail=force_fail)
        elif name == 'memory_store':
            r = memory_store(inputs['key'], inputs['value'], fail=force_fail)
        elif name == 'memory_retrieve':
            r = memory_retrieve(inputs['key'])
        elif name == 'calculator':
            r = calculator(inputs['expression'], fail=force_fail)
        else:
            r = {'status': 'error', 'content': f'Unknown tool: {name}'}
        return json.dumps(r)
    except Exception as e:
        return json.dumps({'status': 'tool_error', 'content': f'{name} failed: {e}'})


def run_agent(user_message, system_prompt, force_fail_tool=None, max_turns=5):
    messages     = [{'role': 'user', 'content': user_message}]
    tools_called = []
    tool_failed  = False
    t0 = time.time()

    for _ in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=1024,
            system=system_prompt, tools=TOOL_DEFINITIONS, messages=messages,
        )

        if resp.stop_reason == 'end_turn':
            text = ''.join(b.text for b in resp.content if hasattr(b, 'text'))
            return {
                'response':     text,
                'tools_called': tools_called,
                'tool_failed':  tool_failed,
                'recovered':    tool_failed,
                'latency_ms':   round((time.time() - t0) * 1000, 1),
            }

        if resp.stop_reason == 'tool_use':
            messages.append({'role': 'assistant', 'content': resp.content})
            tool_results = []
            for block in resp.content:
                if block.type != 'tool_use':
                    continue
                tools_called.append(block.name)
                should_fail = (force_fail_tool == block.name)
                raw = execute_tool(block.name, block.input, force_fail=should_fail)
                rd  = json.loads(raw)
                if rd.get('status') in ('tool_error', 'error'):
                    tool_failed = True
                tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': block.id,
                    'content': rd['content'],
                })
            messages.append({'role': 'user', 'content': tool_results})

    return {
        'response':     '[Agent reached max turns without completing]',
        'tools_called': tools_called,
        'tool_failed':  tool_failed,
        'recovered':    tool_failed,
        'latency_ms':   round((time.time() - t0) * 1000, 1),
    }


# ═══════════════════════════════════════════════════════════════
#  EVAL PROMPTS
# ═══════════════════════════════════════════════════════════════

EVAL_PROMPTS = [
    # Happy path
    {'id':'H01','category':'happy_path','prompt':'What is 347 multiplied by 89?','correct_tool':'calculator','force_fail_tool':None},
    {'id':'H02','category':'happy_path','prompt':'Please remember that my monthly budget is £1,200.','correct_tool':'memory_store','force_fail_tool':None},
    {'id':'H03','category':'happy_path','prompt':'What was the monthly budget I told you?','correct_tool':'memory_retrieve','force_fail_tool':None},
    {'id':'H04','category':'happy_path','prompt':'Search for the latest news about Anthropic AI.','correct_tool':'web_search','force_fail_tool':None},
    {'id':'H05','category':'happy_path','prompt':'What is the square root of 1764?','correct_tool':'calculator','force_fail_tool':None},
    {'id':'H06','category':'happy_path','prompt':'Store the fact that my project deadline is 31st May 2026.','correct_tool':'memory_store','force_fail_tool':None},
    {'id':'H07','category':'happy_path','prompt':'What is 18% of £4,250?','correct_tool':'calculator','force_fail_tool':None},
    {'id':'H08','category':'happy_path','prompt':'Search for the definition of Information Coefficient in quantitative finance.','correct_tool':'web_search','force_fail_tool':None},
    {'id':'H09','category':'happy_path','prompt':'Calculate 2 to the power of 32.','correct_tool':'calculator','force_fail_tool':'calculator'},
    {'id':'H10','category':'happy_path','prompt':'Search for the current price of Bitcoin in USD.','correct_tool':'web_search','force_fail_tool':'web_search'},
    # Ambiguous
    {'id':'A01','category':'ambiguous','prompt':'What is the GDP of France?','correct_tool':'web_search','preferred_tool':'web_search','force_fail_tool':None},
    {'id':'A02','category':'ambiguous','prompt':'How much is 20% off my budget?','correct_tool':'memory_retrieve','preferred_tool':'memory_retrieve','force_fail_tool':None},
    {'id':'A03','category':'ambiguous','prompt':'What is my project deadline?','correct_tool':'memory_retrieve','preferred_tool':'memory_retrieve','force_fail_tool':None},
    {'id':'A04','category':'ambiguous','prompt':'What is the current USD to GBP exchange rate, and how much is £500 in dollars?','correct_tool':'web_search','preferred_tool':'web_search','force_fail_tool':None},
    {'id':'A05','category':'ambiguous','prompt':'Tell me about the SAM optimiser.','correct_tool':'web_search','preferred_tool':'web_search','force_fail_tool':None},
    # Out of scope
    {'id':'O01','category':'out_of_scope','prompt':'Write me a poem about the ocean.','correct_tool':None,'force_fail_tool':None},
    {'id':'O02','category':'out_of_scope','prompt':'Can you book a restaurant for me in Knightsbridge for tonight?','correct_tool':None,'force_fail_tool':None},
    {'id':'O03','category':'out_of_scope','prompt':'What will the FTSE 100 close at tomorrow?','correct_tool':None,'force_fail_tool':None},
    {'id':'O04','category':'out_of_scope','prompt':'Translate this into Mandarin: the quick brown fox.','correct_tool':None,'force_fail_tool':None},
    {'id':'O05','category':'out_of_scope','prompt':'Can you send an email to my manager saying I will be late?','correct_tool':None,'force_fail_tool':None},
]


def grade(result, spec):
    cat          = spec['category']
    tools_called = result['tools_called']
    first_tool   = tools_called[0] if tools_called else None
    preferred    = spec.get('preferred_tool', spec.get('correct_tool'))

    if cat == 'out_of_scope':
        abstained = len(tools_called) == 0
        return {'correct': abstained, 'abstained': abstained}
    else:
        correct = (first_tool == preferred)
        return {'correct': correct, 'abstained': len(tools_called) == 0}


# ═══════════════════════════════════════════════════════════════
#  RUN EVALUATION
# ═══════════════════════════════════════════════════════════════

all_rows = []

for prompt_label, system_prompt in SYSTEM_PROMPTS.items():
    print(f'\n{"="*55}')
    print(f'Running: {prompt_label}')
    print(f'{"="*55}')
    clear_memory()

    for spec in EVAL_PROMPTS:
        short = spec['prompt'][:55] + '...' if len(spec['prompt']) > 55 else spec['prompt']
        print(f'  [{spec["id"]}] {short}')

        result  = run_agent(
            user_message    = spec['prompt'],
            system_prompt   = system_prompt,
            force_fail_tool = spec.get('force_fail_tool'),
        )
        grading = grade(result, spec)

        status    = '✅' if grading['correct'] else '❌'
        fail_note = ' [tool failed → recovered]' if result['tool_failed'] else ''
        print(f'       {status} tools={result["tools_called"]} | {result["latency_ms"]}ms{fail_note}')

        all_rows.append({
            'prompt_id':       spec['id'],
            'category':        spec['category'],
            'prompt':          spec['prompt'],
            'correct_tool':    spec.get('correct_tool'),
            'force_fail':      spec.get('force_fail_tool') is not None,
            'system_prompt':   prompt_label,
            'tools_called':    result['tools_called'],
            'first_tool':      result['tools_called'][0] if result['tools_called'] else None,
            'tool_failed':     result['tool_failed'],
            'recovered':       result['recovered'],
            'latency_ms':      result['latency_ms'],
            'response':        result['response'],
            'correct':         grading['correct'],
            'abstained':       grading['abstained'],
        })

df = pd.DataFrame(all_rows)
df.to_csv('results/raw_eval_results.csv', index=False)
print(f'\nSaved results/raw_eval_results.csv ({len(df)} rows)')


# ═══════════════════════════════════════════════════════════════
#  METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════

def metrics(df, label):
    happy = df[(df['category'] == 'happy_path') & ~df['force_fail']]
    ambig = df[df['category'] == 'ambiguous']
    oos   = df[df['category'] == 'out_of_scope']
    fails = df[df['force_fail']]
    return {
        'System Prompt':           label,
        'Happy Path Accuracy (%)': round(happy['correct'].mean() * 100, 1),
        'Ambiguous Accuracy (%)':  round(ambig['correct'].mean() * 100, 1),
        'OOS Abstention Rate (%)': round(oos['abstained'].mean() * 100, 1),
        'Tool Fail Recovery (%)':  round(fails['recovered'].mean() * 100, 1),
        'Mean Latency (ms)':       round(df['latency_ms'].mean(), 0),
        'P95 Latency (ms)':        round(df['latency_ms'].quantile(0.95), 0),
        'Max-Turns Failures':      df['response'].str.contains('max turns', na=False).sum(),
    }

mdf = pd.DataFrame([
    metrics(df[df['system_prompt'] == lbl], lbl)
    for lbl in df['system_prompt'].unique()
]).set_index('System Prompt')

mdf.to_csv('results/metrics_summary.csv')
print('Saved results/metrics_summary.csv')

print('\n' + '='*55)
print('RESULTS SUMMARY')
print('='*55)
print(mdf.to_string())


# ═══════════════════════════════════════════════════════════════
#  CHART
# ═══════════════════════════════════════════════════════════════

PALETTE = ['#00e5a0', '#4d9aff', '#ff6b6b', '#ffd166']
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'text.color': '#c9d1d9', 'grid.color': '#21262d',
    'grid.alpha': 0.5, 'font.size': 11,
})

prompts = df['system_prompt'].unique()
colors  = PALETTE[:2]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1 — Accuracy by category
ax = axes[0, 0]
cats      = ['happy_path', 'ambiguous', 'out_of_scope']
cat_names = ['Happy Path', 'Ambiguous', 'Out-of-Scope']
x, w = range(len(cats)), 0.35
for i, (prompt, color) in enumerate(zip(prompts, colors)):
    sub  = df[df['system_prompt'] == prompt]
    accs = []
    for cat in cats:
        cdf = sub[sub['category'] == cat]
        accs.append(
            cdf['abstained'].mean() * 100 if cat == 'out_of_scope'
            else cdf['correct'].mean() * 100
        )
    offset = (i - 0.5) * w
    bars = ax.bar([xi + offset for xi in x], accs, w,
                  label=prompt, color=color, alpha=0.85)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(cat_names)
ax.set_ylabel('Accuracy / Abstention (%)')
ax.set_title('Performance by Category')
ax.set_ylim(0, 115)
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.4)

# 2 — Latency distribution
ax = axes[0, 1]
for prompt, color in zip(prompts, colors):
    lat = df[df['system_prompt'] == prompt]['latency_ms']
    ax.hist(lat, bins=12, alpha=0.55, color=color, label=prompt, density=True)
    ax.axvline(lat.mean(), color=color, lw=2, linestyle='--', alpha=0.8)
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Density')
ax.set_title('Response Latency Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# 3 — Tool call frequency
ax = axes[1, 0]
tool_counts = (
    df.explode('tools_called')
    .groupby(['system_prompt', 'tools_called'])
    .size().unstack(fill_value=0)
)
tool_counts.T.plot(kind='bar', ax=ax, color=colors[:len(prompts)],
                   alpha=0.85, edgecolor='none')
ax.set_title('Tool Call Frequency by Prompt')
ax.set_xlabel('Tool')
ax.set_ylabel('Calls')
ax.tick_params(axis='x', rotation=15)
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.4)

# 4 — Correct vs incorrect
ax = axes[1, 1]
summary = df.groupby('system_prompt')['correct'].agg(['sum', 'count'])
summary['incorrect'] = summary['count'] - summary['sum']
xi = range(len(summary))
ax.bar(xi, summary['sum'],       0.4, label='Correct',   color=PALETTE[0], alpha=0.85)
ax.bar(xi, summary['incorrect'], 0.4, label='Incorrect', color=PALETTE[2],
       alpha=0.85, bottom=summary['sum'])
for i, (_, row) in enumerate(summary.iterrows()):
    ax.text(i, row['count'] + 0.2, f'{row["sum"]/row["count"]*100:.0f}%',
            ha='center', fontsize=11)
ax.set_xticks(list(xi))
ax.set_xticklabels([p[:20] for p in summary.index], rotation=10)
ax.set_ylabel('Prompts')
ax.set_title('Total Correct vs Incorrect')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('results/eval_results.png', dpi=150, bbox_inches='tight')
print('Saved results/eval_results.png')

print('\n✅ All done. See results/ folder.')
