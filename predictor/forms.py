from django import forms

TEAM_SIZE = 5


class MatchForm(forms.Form):
    """Ten FACEIT nicknames: five for team A, five for team B."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for side in ("a", "b"):
            for i in range(1, TEAM_SIZE + 1):
                self.fields[f"{side}{i}"] = forms.CharField(
                    max_length=64,
                    strip=True,
                    widget=forms.TextInput(
                        attrs={"placeholder": f"Team {side.upper()} player {i}"}
                    ),
                )

    def _team(self, side: str) -> list[str]:
        return [self.cleaned_data[f"{side}{i}"] for i in range(1, TEAM_SIZE + 1)]

    def team_a(self) -> list[str]:
        return self._team("a")

    def team_b(self) -> list[str]:
        return self._team("b")

    # Convenience for the template (no-arg so it can be called in templates).
    def team_a_fields(self):
        return [self[f"a{i}"] for i in range(1, TEAM_SIZE + 1)]

    def team_b_fields(self):
        return [self[f"b{i}"] for i in range(1, TEAM_SIZE + 1)]
