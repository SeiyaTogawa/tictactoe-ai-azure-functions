import json, os, pathlib
import azure.functions as func
from inference import play_turn_ex


os.environ["TTT_PKL"] = str(pathlib.Path(__file__).with_name("mc_tictactoe.pkl"))

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="playturn", methods=["POST"])
def playturn(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data  = req.get_json()
        board = data["board"]
        move  = int(data["move"])

        result = play_turn_ex(board, move)
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json; charset=utf-8"
        )
    except Exception as e:
        return func.HttpResponse(f"Error: {e}", status_code=400)
