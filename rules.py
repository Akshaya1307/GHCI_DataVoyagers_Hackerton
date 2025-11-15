import yaml

class RuleEngine:
    def __init__(self, rules_path="rules.yaml"):
        with open(rules_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.rules = []
        for entry in data.get("rules", []):
            self.rules.append({
                "name": entry.get("name"),
                "patterns": [p.lower() for p in entry.get("patterns", [])],
                "category": entry.get("category")
            })

    def predict(self, text: str):
        """
        ALWAYS returns exactly TWO values:
        (category, details_dict)
        OR
        (None, None)
        """
        text = text.lower()

        # Loop through rules
        for rule in self.rules:
            for pattern in rule["patterns"]:
                if pattern in text:
                    return (
                        rule["category"],
                        {
                            "rule_name": rule["name"],
                            "matched_pattern": pattern,
                            "category": rule["category"]
                        }
                    )

        # If NO RULE matched
        return None, None
