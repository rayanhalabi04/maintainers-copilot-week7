from fastapi import APIRouter, Depends, HTTPException, Response

from app.domain.widget import WidgetConfig
from app.services.widget_service import WidgetConfigNotFoundError, WidgetService


router = APIRouter(tags=["widget"])

_widget_service = WidgetService()


def get_widget_service() -> WidgetService:
    return _widget_service


@router.get("/widget/config/{widget_id}", response_model=WidgetConfig)
def get_widget_config(
    widget_id: str,
    service: WidgetService = Depends(get_widget_service),
) -> WidgetConfig:
    try:
        return service.get_config(widget_id)
    except WidgetConfigNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Widget config not found.") from exc


@router.get("/widget.js")
def widget_loader(service: WidgetService = Depends(get_widget_service)) -> Response:
    return Response(
        content=service.build_loader_script(),
        media_type="application/javascript",
    )
