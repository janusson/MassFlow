import yogimass
import yogimass.cli
import yogimass.config

def test_version():
    assert yogimass.__version__ == "0.1.0"

def test_imports():
    assert yogimass.cli
    assert yogimass.config
