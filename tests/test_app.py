"""App-builds smoke test (NiceGUI User fixture: no browser, runs webapp/main.py in-process)."""
from nicegui.testing import User


async def test_app_builds_and_projects(user: User, index) -> None:
    # `index` (session fixture) warms the player index, so the app's startup load is instant.
    await user.open("/")
    await user.should_see(marker="search", retries=100)
    await user.should_see(marker="injury-note")
    await user.should_see(marker="model-info")

    user.find(marker="search").type("mahomes")
    await user.should_see(marker="suggestion", retries=100)
    user.find(marker="suggestion").click()
    await user.should_see(marker="result-card", retries=200)
    await user.should_see(marker="game-header")
    await user.should_see(marker="stat-line")
    await user.should_see("Pass yds")

    user.find(marker="dev-toggle").click()
    await user.should_see(marker="dev-details", retries=50)
