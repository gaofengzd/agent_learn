"""FastAPI/Jinja local web shell. Business work is delegated to a UI facade."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Protocol
from fastapi import FastAPI,Request,UploadFile,File,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

class UIFacade(Protocol):
    def navigation(self)->dict[str,list[dict[str,str]]]:...
    def health(self)->list[dict[str,str]]:...

class EmptyUIFacade:
    def navigation(self):return {"papers":[],"conversations":[]}
    def health(self):return [{"name":"系统","status":"warning","message":"服务尚未配置"}]
    def list_papers(self):return []
    def qa_view(self):return {}
    def analysis_view(self):return {}
    def upload_papers(self,files):raise RuntimeError("论文处理服务尚未配置")
    def delete_paper(self,paper_id):raise RuntimeError("论文处理服务尚未配置")
    def retry_paper(self,paper_id):raise RuntimeError("论文处理服务尚未配置")
    def reprocess_paper(self,paper_id):raise RuntimeError("论文处理服务尚未配置")

@dataclass(frozen=True,slots=True)
class UIState:
    level:str;icon:str;label:str;message:str

def create_app(facade:UIFacade|None=None)->FastAPI:
    facade=facade or EmptyUIFacade();base=Path(__file__).parent
    app=FastAPI(title="论文精读 Agent",docs_url=None,redoc_url=None)
    app.state.facade=facade
    app.mount("/static",StaticFiles(directory=base/"static"),name="static")
    templates=Jinja2Templates(directory=base/"templates")
    @app.get("/",response_class=HTMLResponse)
    def home(request:Request,section:str="read"):
        try:
            navigation=facade.navigation();health=facade.health();state=None
        except Exception as exc:
            navigation={"papers":[],"conversations":[]};health=[]
            state=UIState("error","!","错误",f"界面数据加载失败：{type(exc).__name__}")
        return templates.TemplateResponse(request=request,name="base.html",context={
            "section":section,"navigation":navigation,"health":health,"state":state,
            "papers":facade.list_papers() if section=="library" and hasattr(facade,"list_papers") else [],
            "qa":facade.qa_view() if section=="read" and hasattr(facade,"qa_view") else {},
            "analysis":facade.analysis_view() if section=="analysis" and hasattr(facade,"analysis_view") else {}})
    def library_response(request:Request,state:UIState|None=None):
        return templates.TemplateResponse(request=request,name="base.html",context={
            "section":"library","navigation":facade.navigation(),"health":facade.health(),
            "state":state,"papers":facade.list_papers()})
    @app.post("/library/upload",response_class=HTMLResponse)
    async def upload(request:Request,files:list[UploadFile]=File(...)):
        try:
            results=facade.upload_papers([(f.filename or "paper.pdf",f.content_type,await f.read()) for f in files])
            duplicates=sum(bool(x.get("duplicate")) for x in results)
            state=UIState("success","✓","成功",f"已接收 {len(results)} 篇论文；重复 {duplicates} 篇")
        except Exception as exc:state=UIState("error","!","错误",f"上传失败：{type(exc).__name__}: {exc}")
        return library_response(request,state)
    @app.post("/library/{paper_id}/delete",response_class=HTMLResponse)
    def delete(request:Request,paper_id:str,confirm:str=Form("")):
        try:
            if confirm!="yes":raise ValueError("必须确认删除")
            facade.delete_paper(paper_id);state=UIState("success","✓","成功","论文已删除")
        except Exception as exc:state=UIState("error","!","错误",f"删除失败：{type(exc).__name__}: {exc}")
        return library_response(request,state)
    @app.post("/library/{paper_id}/{action}",response_class=HTMLResponse)
    def process_action(request:Request,paper_id:str,action:str):
        try:
            if action=="retry":facade.retry_paper(paper_id)
            elif action=="reprocess":facade.reprocess_paper(paper_id)
            else:raise ValueError("未知操作")
            state=UIState("success","✓","成功","处理任务已提交")
        except Exception as exc:state=UIState("error","!","错误",f"操作失败：{type(exc).__name__}: {exc}")
        return library_response(request,state)
    def qa_response(request:Request,state:UIState|None=None):
        return templates.TemplateResponse(request=request,name="base.html",context={
            "section":"read","navigation":facade.navigation(),"health":facade.health(),
            "state":state,"papers":[],"qa":facade.qa_view()})
    @app.post("/sessions",response_class=HTMLResponse)
    def create_session(request:Request,title:str=Form(...)):
        try:facade.create_session(title);state=UIState("success","✓","成功","会话已创建")
        except Exception as exc:state=UIState("error","!","错误",f"创建失败：{type(exc).__name__}: {exc}")
        return qa_response(request,state)
    @app.post("/sessions/{conversation_id}/delete",response_class=HTMLResponse)
    def delete_session(request:Request,conversation_id:str):
        try:facade.delete_session(conversation_id);state=UIState("success","✓","成功","会话已删除")
        except Exception as exc:state=UIState("error","!","错误",f"删除失败：{type(exc).__name__}: {exc}")
        return qa_response(request,state)
    @app.post("/sessions/{conversation_id}/scope",response_class=HTMLResponse)
    def change_scope(request:Request,conversation_id:str,scope_mode:str=Form(...),paper_ids:list[str]=Form([])):
        try:facade.change_scope(conversation_id,scope_mode,paper_ids);state=UIState("warning","△","范围已变化","后续回答只使用新范围")
        except Exception as exc:state=UIState("error","!","错误",f"范围切换失败：{type(exc).__name__}: {exc}")
        return qa_response(request,state)
    @app.post("/sessions/{conversation_id}/ask",response_class=HTMLResponse)
    def ask(request:Request,conversation_id:str,question:str=Form(...)):
        try:facade.ask(conversation_id,question);state=UIState("success","✓","完成","回答已生成")
        except Exception as exc:state=UIState("error","!","错误",f"问答失败：{type(exc).__name__}: {exc}")
        return qa_response(request,state)
    def analysis_response(request:Request,state:UIState|None=None):
        return templates.TemplateResponse(request=request,name="base.html",context={
            "section":"analysis","navigation":facade.navigation(),"health":facade.health(),
            "state":state,"papers":[],"qa":{},"analysis":facade.analysis_view()})
    @app.post("/analysis/{operation}",response_class=HTMLResponse)
    def run_analysis(request:Request,operation:str,paper_ids:list[str]=Form([]),level:str=Form("standard")):
        try:
            if operation=="summary":facade.summarize(paper_ids,level)
            elif operation=="methods":facade.extract_methods(paper_ids)
            elif operation=="innovations":facade.analyze_innovations(paper_ids)
            else:raise ValueError("未知分析类型")
            state=UIState("success","✓","完成","阅读分析已生成")
        except Exception as exc:state=UIState("error","!","错误",f"分析失败：{type(exc).__name__}: {exc}")
        return analysis_response(request,state)
    return app
