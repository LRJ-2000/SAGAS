import random
from tqdm import tqdm
from pathlib import Path
from absl import app, flags
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))
from L_utils.ltl_utils import check_formula_validity


# ---------- Flags ----------
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "total_num_regions", 8, "Total number of regions in the environment"
)
flags.DEFINE_integer("combine_max_templates", 4, "Max number of templates to combine")
flags.DEFINE_integer("formula_num", 10000, "Number of formulas to generate")
flags.DEFINE_integer(
    "region_num_limit", -1, "Max number of regions in a formula, -1 for no limit"
)
flags.DEFINE_string(
    "complexity",
    "hard",
    "Complexity level of formulas: easy, medium, hard",
)

# ---------- LTL Formula Template ----------
templates = [
    # Visit region eventually
    {"pattern": "F({0})", "num_regions": 1, "complexity": "easy"},
    # Never enter region
    {"pattern": "G(!{0})", "num_regions": 1, "complexity": "easy"},
    # Visit a followed by b
    {"pattern": "F({0} & F({1}))", "num_regions": 2, "complexity": "easy"},
    {"pattern": "F({0} & FG({1}))", "num_regions": 2, "complexity": "easy"},
    # Infinitely revisit
    {"pattern": "G(F({0}))", "num_regions": 1, "complexity": "easy"},
    {"pattern": "G(F({0} & F({1})))", "num_regions": 2, "complexity": "medium"},
    {
        "pattern": "G(F({0} & F({1} & F({2}))))",
        "num_regions": 3,
        "complexity": "medium",
    },
    {
        "pattern": "G(F({0} & F({1} & F({2} & F({3})))))",
        "num_regions": 4,
        "complexity": "hard",
    },
    # Eventually visit a or b
    {"pattern": "F({0} | {1})", "num_regions": 2, "complexity": "easy"},
    {"pattern": "F({0} | {1} | {2})", "num_regions": 3, "complexity": "easy"},
    {"pattern": "F({0} | {1} | {2} | {3})", "num_regions": 4, "complexity": "medium"},
    # always eventually a and eventually b
    {"pattern": "G(F({0}) & F({1}))", "num_regions": 2, "complexity": "easy"},
    {
        "pattern": "G(F({0}) & F({1}) & F({2}))",
        "num_regions": 3,
        "complexity": "medium",
    },
    {
        "pattern": "G(F({0}) & F({1}) & F({2}) & F({3}))",
        "num_regions": 4,
        "complexity": "hard",
    },
    # Visit a, b, c in sequence
    {"pattern": "F({0} & F({1} & F({2})))", "num_regions": 3, "complexity": "medium"},
    {"pattern": "F({0} & F({1} & FG({2})))", "num_regions": 3, "complexity": "medium"},
    # Visit a, b, c, d in sequence
    {
        "pattern": "F({0} & F({1} & F({2} & F({3}))))",
        "num_regions": 4,
        "complexity": "medium",
    },
    {
        "pattern": "F({0} & F({1} & F({2} & FG({3}))))",
        "num_regions": 4,
        "complexity": "medium",
    },
    # After a, avoid c until b
    {"pattern": "F({0} & (!{1} U {2}))", "num_regions": 3, "complexity": "medium"},
    # F(b) ∧ (¬b U a)
    {"pattern": "F({0}) & (!{0} U {1})", "num_regions": 2, "complexity": "easy"},
    # Eventually stay forever
    {"pattern": "FG({0})", "num_regions": 1, "complexity": "easy"},
    # Avoid until visited
    {"pattern": "!{0} U {1}", "num_regions": 2, "complexity": "easy"},
    # Visit all three any order
    {"pattern": "F({0}) & F({1}) & F({2})", "num_regions": 3, "complexity": "easy"},
    # Visit all four any order
    {"pattern": "F({0}) & F({1}) & F({2}) & F({3})", "num_regions": 4, "complexity": "medium"},
    # Strict sequence
    {
        "pattern": "F({0}) & F({1}) & F({2}) & (!{1} U {0}) & (!{2} U {1})",
        "num_regions": 3,
        "complexity": "medium",
    },
    # Visit a region last
    {
        "pattern": "F({0}) & F({1}) & F({2}) & (!{0} U {1}) & (!{0} U {2})",
        "num_regions": 3,
        "complexity": "medium",
    },
    # visit a, b, c, d, e in sequence
    {
        "pattern": "F({0} & F({1} & F({2} & F({3} & F({4})))))",
        "num_regions": 5,
        "complexity": "hard",
    },
    # Visit a, b, c, d, e, f in sequence
    {
        "pattern": "F({0} & F({1} & F({2} & F({3} & F({4} & F({5}))))))",
        "num_regions": 6,
        "complexity": "hard",
    },
    # Visit all five any order
    {
        "pattern": "F({0}) & F({1}) & F({2}) & F({3}) & F({4})",
        "num_regions": 5,
        "complexity": "medium",
    },
    # Visit all six any order
    {
        "pattern": "F({0}) & F({1}) & F({2}) & F({3}) & F({4}) & F({5})",
        "num_regions": 6,
        "complexity": "hard",
    },
    # Strict sequence
    {
        "pattern": "F({0}) & F({1}) & F({2}) & F({3}) & (!{1} U {0}) & (!{2} U {1}) & (!{3} U {2})",
        "num_regions": 4,
        "complexity": "hard",
    },
    # Visit a region last
    {
        "pattern": "F({0}) & F({1}) & F({2}) & F({3}) & (!{0} U {1}) & (!{0} U {2}) & (!{0} U {3})",
        "num_regions": 4,
        "complexity": "hard",
    },
]


def generate_single_formula(template, regions):
    """Generate an LTL formula from a single template"""
    num = template["num_regions"]

    selected = random.sample(regions, num)

    return template["pattern"].format(*selected), num


def generate_combined_formula(regions):
    """Generate a formula combining up to 2 templates"""
    # Filter templates based on complexity level
    complexity_level = FLAGS.complexity
    assert complexity_level in ["easy", "medium", "hard"], "Invalid complexity level, must be easy, medium, or hard"
    if complexity_level == "easy":
        filtered_templates = [t for t in templates if t["complexity"] == "easy"]
    elif complexity_level == "medium":
        filtered_templates = [
            t for t in templates if t["complexity"] in ["easy", "medium"]
        ]
    else:  # hard
        filtered_templates = templates
    # Further filter templates based on available regions
    filtered_templates = [t for t in templates if t["num_regions"] <= len(regions)]

    # Combine templates
    while True:
        num_templates = random.choice(range(1, FLAGS.combine_max_templates + 1))
        selected_templates = random.sample(filtered_templates, num_templates)
        region_limit = FLAGS.region_num_limit
        assert (
            region_limit > 0 or region_limit == -1
        ), "region_num_limit must be -1 or positive"
        if region_limit == -1:
            region_limit = np.inf

        parts = []
        total_regions = 0
        for t in selected_templates:
            formula, num = generate_single_formula(t, regions)
            total_regions += num
            if total_regions > region_limit:  # Limit complexity
                break
            parts.append(f"{formula}")
        if len(parts) == 0:
            continue
        return " & ".join(parts)


def main(_):
    regions = [f"p{i}" for i in range(FLAGS.total_num_regions)]
    formula_set = set()
    cnt_last = 0
    duplicate_count = 0
    with tqdm(total=FLAGS.formula_num) as pbar:
        while True:
            formula = generate_combined_formula(regions)
            labels = [
                f"p{i}" for i in range(FLAGS.total_num_regions) if f"p{i}" in formula
            ]
            if not check_formula_validity(formula, labels):
                continue
            formula_set.add(formula)
            cnt_current = len(formula_set)
            if cnt_current == cnt_last:
                duplicate_count += 1
                if duplicate_count >= 1000:
                    print(
                        f"Warning: Stopped early due to too many duplicates. Generated {cnt_current} formulas."
                    )
                    break
            else:
                duplicate_count = 0

            if cnt_current > cnt_last:
                pbar.update(1)
                cnt_last = cnt_current
            if cnt_current >= FLAGS.formula_num:
                break

    output_dir = Path(__file__).parent / "formula"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_filepath = output_dir / "formula_hard"
    # save_filepath = output_dir / FLAGS.complexity
    save_filepath = save_filepath.with_suffix(".txt")
    with open(save_filepath, "w") as f:
        for formula in formula_set:
            f.write(formula + "\n")


if __name__ == "__main__":
    app.run(main)
