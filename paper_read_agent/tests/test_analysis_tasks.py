from threading import Event
import time

from paper_read_agent.persistence.database import SQLiteDatabase
from paper_read_agent.ui.local_facade import _AnalysisTaskQueue


def database(tmp_path):
    value=SQLiteDatabase(tmp_path/"tasks.sqlite3")
    value.initialize()
    return value


def test_analysis_queue_runs_work_outside_submit_and_reports_success(tmp_path):
    release = Event()
    calls = []

    def execute(operation, paper_ids, level, cancellation_check):
        calls.append((operation, paper_ids, level))
        release.wait(2)

    queue = _AnalysisTaskQueue(execute,database(tmp_path))
    task = queue.submit("summary", ("p1",), ("v1",), "detailed")
    assert task["status"] == "queued"
    assert queue.future(task["task_id"]) is not None
    release.set()
    queue.future(task["task_id"]).result(timeout=2)
    result = queue.snapshot()[0]
    queue.shutdown()

    assert calls == [("summary", ("p1",), "detailed")]
    assert result["status"] == "succeeded"


def test_analysis_queue_exposes_failure_without_raising_in_http_thread(tmp_path):
    def execute(*args):
        raise RuntimeError("glm unavailable")

    queue = _AnalysisTaskQueue(execute,database(tmp_path))
    task = queue.submit("methods", ("p1",), ("v1",))
    queue.future(task["task_id"]).result(timeout=2)
    result = queue.snapshot()[0]
    queue.shutdown()

    assert result["status"] == "failed"
    assert result["message"] == "RuntimeError: glm unavailable"


def test_analysis_result_survives_queue_recreation(tmp_path):
    db=database(tmp_path)
    first=_AnalysisTaskQueue(
        lambda *args:{"analysis":{"summary":{"content":"restored"}},
                     "citations":[{"evidence_id":"ev1"}]},db)
    task=first.submit("summary",("p1",),("v1",),"brief")
    first.future(task["task_id"]).result(timeout=2)
    first.shutdown()

    second=_AnalysisTaskQueue(lambda *args:None,db)
    assert second.successful_payloads()[0]["analysis"]["summary"]["content"]=="restored"
    assert second.successful_payloads()[0]["_version_ids"]==["v1"]
    second.shutdown()


def test_queue_recreation_marks_interrupted_tasks_failed(tmp_path):
    db=database(tmp_path)
    now="2026-01-01T00:00:00+00:00"
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO analysis_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("t","summary",'["p1"]','["v1"]',"brief","running",
             "正在生成",0,None,now,now,None,now)
        )

    queue=_AnalysisTaskQueue(lambda *args:None,db)
    task=queue.snapshot()[0]
    queue.shutdown()

    assert task["status"]=="failed"
    assert "服务重启" in task["message"]


def test_duplicate_active_analysis_reuses_existing_task(tmp_path):
    release=Event()
    started=Event()
    def execute(*args):
        started.set()
        release.wait(2)
        return {}
    queue=_AnalysisTaskQueue(execute,database(tmp_path))
    first=queue.submit("summary",("p1",),("v1",),"brief")
    started.wait(1)
    duplicate=queue.submit("summary",("p1",),("v1",),"brief")
    release.set()
    queue.future(first["task_id"]).result(timeout=2)
    queue.shutdown()

    assert duplicate["duplicate"] is True
    assert duplicate["task_id"]==first["task_id"]


def test_queued_task_can_be_cancelled_before_execution(tmp_path):
    release=Event()
    started=Event()
    def execute(*args):
        started.set()
        release.wait(2)
        return {}
    queue=_AnalysisTaskQueue(execute,database(tmp_path))
    first=queue.submit("summary",("p1",),("v1",),"brief")
    started.wait(1)
    queued=queue.submit("methods",("p1",),("v1",))
    cancelled=queue.cancel(queued["task_id"])
    release.set()
    queue.future(first["task_id"]).result(timeout=2)
    queue.shutdown()

    assert cancelled["status"]=="cancelled"


def test_running_task_observes_cooperative_cancellation(tmp_path):
    started=Event()
    def execute(operation,paper_ids,level,cancellation_check):
        started.set()
        deadline=time.time()+2
        while not cancellation_check() and time.time()<deadline:
            time.sleep(.01)
        raise RuntimeError("Analysis cancelled")
    queue=_AnalysisTaskQueue(execute,database(tmp_path))
    task=queue.submit("methods",("p1",),("v1",))
    started.wait(1)
    queue.cancel(task["task_id"])
    queue.future(task["task_id"]).result(timeout=2)
    result=queue.snapshot_task(task["task_id"])
    queue.shutdown()

    assert result["status"]=="cancelled"
