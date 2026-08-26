import json
import time
from pathlib import Path

import pytest

from paper_read_agent.config import ChunkSettings
from paper_read_agent.document_pipeline.chunking import ParentChildChunker
from paper_read_agent.document_pipeline.normalizer import NormalizedDocument
from paper_read_agent.domain.models import ContentBlock, Page

ROOT=Path(__file__).parent/"e2e"

def cases():return json.loads((ROOT/"acceptance_cases.json").read_text(encoding="utf-8"))

def test_twenty_fixed_acceptance_cases_cover_required_corpus_and_failure_layers():
    values=cases()
    assert len(values)==20 and [x["id"] for x in values]==[f"A{i:02d}" for i in range(1,21)]
    fixtures={x["fixture"] for x in values};layers={x["layer"] for x in values}
    assert {"native_zh","native_en","scanned_zh","mixed_pdf","two_column","table_formula","conflict","versions","low_quality_scan"}<=fixtures
    assert layers=={"parsing","retrieval","generation","citation"}
    assert all(x["question"] and x["scope"] and isinstance(x["evidence"],list) for x in values)

@pytest.mark.parametrize("case",cases(),ids=lambda x:x["id"])
def test_each_acceptance_case_has_a_diagnosable_expected_layer(case):
    assert case["layer"] in {"parsing","retrieval","generation","citation"}
    if not case["evidence"]:assert case["id"] in {"A11","A19"}

class Tokenizer:
    def encode(self,text,*,add_special_tokens=False):return list(text)
    def decode(self,ids,*,skip_special_tokens=True):return "".join(ids)

@pytest.mark.performance
def test_chunking_smoke_stays_within_recorded_local_baseline():
    page=Page("opaque","v",1)
    blocks=tuple(ContentBlock(f"b{i}","v","opaque",("Methods",),"text",f"第{i}段 method evidence") for i in range(200))
    document=NormalizedDocument("v",(page,),blocks)
    settings=ChunkSettings(20,100,100,300,0.1)
    started=time.perf_counter();result=ParentChildChunker(settings,tokenizer=Tokenizer()).build(document)
    assert result.chunks and time.perf_counter()-started<5

def test_performance_baseline_is_versioned_with_the_suite():
    text=(ROOT/"PERFORMANCE_BASELINE.md").read_text(encoding="utf-8")
    assert "20 个 JSON 案例" in text and "< 5 秒" in text and "不是产品 SLA" in text
