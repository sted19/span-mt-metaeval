# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import re
from collections import defaultdict
import logging

from mt_evaluation.autoevals.autoeval import AutoEval
from mt_evaluation.core import Prompt, AutomaticEvaluation, Error


logger = logging.getLogger(__name__)


class GembaMQM(AutoEval):
    prompt = Prompt(
        system_prompt="You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.",
        user_prompt="""{src_lang} source:
```{src}```
{tgt_lang} translation:
```{tgt}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension.""",
        few_shots=[
            {
                "src_lang": "English",
                "src": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
                "tgt_lang": "German",
                "tgt": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
                "response": """Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
            },
            {
                "src_lang": "English",
                "src": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
                "tgt_lang": "Czech",
                "tgt": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
                "response": """Critical:
no-error
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""",
            },
            {
                "src_lang": "Chinese",
                "src": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
                "tgt_lang": "English",
                "tgt": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
                "response": """Critical:
accuracy/addition - "of high-speed rail"
Major:
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""",
            },
        ],
    )

    @staticmethod
    def parse_broken_json(x):
        improved_translation = ""
        errors = defaultdict(list)
        if '"errors": ' in x and "improved translation" in x:
            data = x.split('", "errors": ')
            if len(data) != 2:
                return {"improved translation": improved_translation, "errors": errors}
            # from data[0] parse improved translation
            improved_translation = data[0].split('"improved translation": "')[1]
            # remove last character from data[1]
            data[1] = data[1][:-1]

            try:
                errors = json.loads(data[1])
            except:
                # just try to get error count
                words = re.findall(r"\b\w+\b", data[1].lower())
                keywords = ["critical", "major", "minor"]

                last_key = None
                for word in words:
                    if word in keywords:
                        last_key = word
                    elif last_key is not None and word == "class":
                        errors[last_key].append({"class": "other"})

        return {"improved translation": improved_translation, "errors": errors}

    @staticmethod
    def parse_error_class(error):
        # parse error from error description, errors are ['accuracy', 'fluency', 'locale convention', 'style', 'terminology', 'non-translation', 'other']
        #  locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use),
        class_name = "unknown"
        if "accuracy" in error:
            class_name = "accuracy"
            for subclass in [
                "addition",
                "mistranslation",
                "omission",
                "untranslated text",
            ]:
                if subclass in error:
                    class_name = f"accuracy-{subclass}"
        elif "fluency" in error:
            class_name = "fluency"
            for subclass in [
                "character encoding",
                "grammar",
                "inconsistency",
                "punctuation",
                "register",
                "spelling",
            ]:
                if subclass in error:
                    class_name = f"fluency-{subclass}"
        elif "locale convention" in error:
            class_name = "locale convention"
            for subclass in ["currency", "date", "name", "telephone", "time"]:
                if subclass in error:
                    class_name = f"locale convention-{subclass}"
        elif "style" in error:
            class_name = "style"
        elif "terminology" in error:
            class_name = "terminology"
            for subclass in ["inappropriate", "inconsistent"]:
                if subclass in error:
                    class_name = f"terminology-{subclass}"
        elif "non-translation" in error:
            class_name = "non-translation"
        elif "other" in error:
            class_name = "other"

        return class_name

    @staticmethod
    def parse_error_span(error) -> str | None:
        """
        Extract text between quotation marks, but only when the opening quote
        is outside parentheses. Parentheses inside quotes are preserved. Finally, return the first span
        # NOTE: we are assuming here that the first reported span is the one most likely to represent the error, in
        case there are more than a single span

        Args:
            error (str): Input string to parse

        Returns:
            str: the extracted error span
        """
        result = []
        i = 0
        paren_depth = 0

        while i < len(error):
            if error[i] == "(":
                paren_depth += 1
            elif error[i] == ")":
                paren_depth -= 1
            elif error[i] == '"' and paren_depth == 0:
                # Found opening quote outside parentheses
                i += 1  # Move past the opening quote
                start = i
                # Find the closing quote (don't care about parentheses inside quotes)
                while i < len(error) and error[i] != '"':
                    i += 1
                if i < len(error):  # Found closing quote
                    result.append(error[start:i])
            i += 1

        return result[0] if result else None

    def parse_response(self, x, full_desc=False):
        if x is None:
            return None

        x = str(x)
        if x.startswith('{"improved translation"'):
            try:
                x = json.loads(x)
            except:
                x = self.parse_broken_json(x)
            errors = x["errors"]

        else:
            x = x.lower()
            errors = {"critical": [], "major": [], "minor": []}
            critical_done, major_done, minor_done = False, False, False
            error_level = None
            for line in x.split("\n"):
                line = line.strip()
                if "no-error" in line or "no error" in line or "" == line:
                    continue
                if "critical:" == line:
                    if not critical_done:
                        error_level = "critical"
                        critical_done = True
                        continue
                    else:
                        break
                elif "major:" == line:
                    if not major_done:
                        error_level = "major"
                        major_done = True
                        continue
                    else:
                        break
                elif "minor:" == line:
                    if not minor_done:
                        error_level = "minor"
                        minor_done = True
                        continue
                    else:
                        break

                if "non-translation" in line:
                    errors["critical"].append(line)
                    continue

                if any(
                    [
                        line.startswith(x)
                        for x in [
                            "accuracy",
                            "fluency",
                            "style",
                            "terminology",
                            "non-translation",
                            "other",
                        ]
                    ]
                ):
                    if error_level is None:
                        print(f"Error level is None for response {x}")
                        continue
                    errors[error_level].append(line)

        error_classes = defaultdict(list)
        final_score = 0
        error_counter = 0
        for error_level in ["critical", "major", "minor"]:
            if error_level not in errors:
                continue
            for error in errors[error_level]:
                if error_counter < 5:
                    final_score += (
                        25
                        if error_level == "critical"
                        else 5 if error_level == "major" else 1
                    )
                    error_counter += 1

                if full_desc:
                    error_classes[error_level].append(error)
                else:
                    class_name = self.parse_error_class(error)
                    error_span = self.parse_error_span(error)
                    error_classes[error_level].append((class_name, error_span))
        if final_score > 25:
            final_score = 25
        # negative score is to normalize that higher score is better
        final_score = -final_score

        full_error_list = []
        for error_level, class_error_list in error_classes.items():
            for error_category, error_span in class_error_list:
                full_error_list.append(
                    Error(
                        span=error_span,
                        category=error_category,
                        severity=error_level,
                        explanation=None,
                    )
                )

        # parsing error set to None for compatibility, but there are not parsing errors in Gemba-mqm (there are actually many, but it is not a simple json parsing so not clear how to detect them)
        return AutomaticEvaluation(
            annotation=x, errors=full_error_list, score=final_score, parsing_error=None
        )
