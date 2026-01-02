from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from src.mnist.mnist_router import router as mnist_router
from src.tictactoe.tictactoe_router import router as tictactoe_router
from src.gomoku.gomoku_router import router as gomoku_router

app = FastAPI(
    title="KMJ AI World API",
    description="API for MNIST digit recognition, Tic-Tac-Toe AI, and Gomoku AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(mnist_router)
app.include_router(tictactoe_router)
app.include_router(gomoku_router)


@app.get("/")
async def root():
    return {
        "message": "KMJ AI World API",
        "endpoints": {
            "mnist": {
                "predict": "/mnist/predict",
                "train": "/mnist/train",
                "test": "/mnist/test",
                "best_config": "/mnist/best-config"
            },
            "tictactoe": {
                "api": "/tictactoe",
                "game": "/static/tictactoe.html",
                "new_game": "/tictactoe/new-game",
                "move": "/tictactoe/move"
            },
            "gomoku": {
                "api": "/gomoku",
                "game": "/static/gomoku.html",
                "new_game": "/gomoku/new-game",
                "move": "/gomoku/move",
                "info": "/gomoku/info"
            },
            "docs": "/docs"
        }
    }