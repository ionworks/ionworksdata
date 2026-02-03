import pytest
import pandas as pd


@pytest.fixture
def gitt_data():
    return {"Synthetic": pd.read_csv("tests/test_data/gitt_synthetic.csv")}


@pytest.fixture
def hppt_data():
    return {"Synthetic": pd.read_csv("tests/test_data/hppt_synthetic.csv")}


@pytest.fixture
def full_rpt_data():
    return {"Synthetic": pd.read_csv("tests/test_data/full_rpt_synthetic.csv")}


@pytest.fixture
def cccv_data():
    return {"Synthetic": pd.read_csv("tests/test_data/cccv_synthetic.csv")}
