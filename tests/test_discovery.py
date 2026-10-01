import pandas as pd
import pytest

from src import data, discovery

needs_data = pytest.mark.skipif(
    not data.is_downloaded(), reason="data/raw/ missing or unverified; run `make data`"
)


def test_top_threshold_counts_ties():
    tc = [150, 140, 130, 130, 120] + [10] * 295  # 300 materials -> top 3
    assert discovery.top_threshold(tc) == 130.0


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        discovery.build_pool(pd.DataFrame(), scenario="easy")


@needs_data
def test_committed_exclusions_match_the_rule():
    saved = pd.read_csv(discovery.EXCLUSIONS, index_col="composition_key")
    fresh = discovery.exclusions(data.load_unique_m())
    assert list(saved.index) == list(fresh.index)
    assert list(saved.suspect) == list(fresh.suspect)


@needs_data
def test_pools_drop_exclusions_and_hard_pool_has_no_cuprates():
    um = data.load_unique_m()
    excluded = set(discovery.exclusions(um).index)
    main, hard = discovery.build_pool(um, "main"), discovery.build_pool(um, "hard")
    assert not excluded & set(main.index)
    assert not excluded & set(hard.index)
    assert (hard.family != data.CUPRATE).all()
    assert set(hard.index) <= set(main.index)
    assert main.index.is_unique
