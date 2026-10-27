"""Direct, offline tests for lightweight variant/component declarations."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import unittest

# Direct execution needs project imports before loading the registry.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.med_copilot_variants import COMPONENTS, VARIANTS, ComponentSpec, Parameter, VariantSpec


class ParameterTests(unittest.TestCase):
    def test_string_boolean_integer_and_number_values_are_preserved(self):
        for kind, values in (("str", ("focused", " café ")), ("bool", (True, False)),
                             ("int", (-2, 0, 3)), ("number", (-2, 0.0, 0.5, 3))):
            parameter = Parameter(kind)
            for value in values:
                with self.subTest(kind=kind, value=value):
                    self.assertEqual(parameter.validate(value, "custom.value"), value)

    def test_wrong_types_and_blank_strings_fail_with_parameter_location(self):
        for kind, values in {
            "str": (None, "", " \t\n", False, 0, [], {}),
            "bool": (None, 0, 1, "true", [], {}),
            "int": (None, True, False, 1.0, "2", [], {}),
            "number": (None, True, False, "0.5", [], {}),
            "unsupported": (0, "value", False, None),
        }.items():
            for value in values:
                with self.subTest(kind=kind, value=value), self.assertRaisesRegex(ValueError, "custom.value"):
                    Parameter(kind).validate(value, "custom.value")

    def test_inclusive_finite_ranges_and_unbounded_integer_values(self):
        for kind in ("int", "number"):
            parameter = Parameter(kind, minimum=0, maximum=3)
            for value in (0, 1, 3):
                self.assertEqual(parameter.validate(value, "limit"), value)
            for value in (-1, 4, float("inf"), float("-inf"), float("nan"), 10 ** 400):
                with self.subTest(kind=kind, value=str(value)), self.assertRaisesRegex(ValueError, "limit"):
                    parameter.validate(value, "limit")
        self.assertEqual(Parameter("int").validate(10 ** 100, "large"), 10 ** 100)

    def test_defaults_shared_field_reference_and_immutability(self):
        parameter = Parameter("int", default=3, config_field="patient_retrieval_top_k", minimum=1)
        self.assertEqual((parameter.kind, parameter.default, parameter.config_field, parameter.minimum),
                         ("int", 3, "patient_retrieval_top_k", 1))
        with self.assertRaises(FrozenInstanceError):
            parameter.default = 8


class DefinitionTests(unittest.TestCase):
    def test_variants_stays_at_top_after_its_required_type_definition(self):
        tree = ast.parse((ROOT / "scripts/med_copilot_variants.py").read_text())
        statements = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                      and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
        self.assertIsInstance(statements[0], ast.ClassDef)
        self.assertEqual(statements[0].name, "VariantSpec")
        self.assertIsInstance(statements[1], ast.Assign)
        self.assertEqual(statements[1].targets[0].id, "VARIANTS")

    def test_component_contract_keeps_resource_credentials_and_implementation_identity(self):
        parameter = Parameter("int", default=4, minimum=1)
        component = ComponentSpec("Custom search", ("evidence",), {"limit": parameter},
                                  ("guidelines",), ("CUSTOM_API_KEY",), "scripts/custom_search.py")
        self.assertEqual(component.description, "Custom search")
        self.assertEqual(component.outputs, ("evidence",))
        self.assertIs(component.parameters["limit"], parameter)
        self.assertEqual(component.resources, ("guidelines",))
        self.assertEqual(component.credentials, ("CUSTOM_API_KEY",))
        self.assertEqual(component.implementation, "scripts/custom_search.py")
        with self.assertRaises(FrozenInstanceError):
            component.outputs = ()

    def test_definition_mapping_defaults_are_independent(self):
        first, second = ComponentSpec("First", ("text",)), ComponentSpec("Second", ("text",))
        first.parameters["limit"] = Parameter("int", 2)
        self.assertEqual(second.parameters, {})
        one, two = VariantSpec("One", (), "llm_only"), VariantSpec("Two", (), "llm_only")
        one.bindings["evidence"] = ("component", "text")
        self.assertEqual(two.bindings, {})

    def test_variant_contract_decouples_prompt_identity_from_component_selection(self):
        variant = VariantSpec("New combination", ("custom_source",), "guidelines",
                              {"Key_info": ("custom_source", "evidence")})
        self.assertEqual(variant.description, "New combination")
        self.assertEqual(variant.components, ("custom_source",))
        self.assertEqual(variant.prompt_key, "guidelines")
        self.assertEqual(variant.bindings, {"Key_info": ("custom_source", "evidence")})
        with self.assertRaises(FrozenInstanceError):
            variant.prompt_key = "llm_only"

    def test_existing_definitions_preserve_order_dependencies_and_prompt_bindings(self):
        expected = {
            "llm_only": ((), {}),
            "patient_retrieval": (("patient_retrieval",), {"example": ("patient_retrieval", "example")}),
            "guidelines": (("guidelines",), {"Key_info": ("guidelines", "Key_info")}),
            "patient_retrieval_guidelines": (("patient_retrieval", "guidelines"), {
                "example": ("patient_retrieval", "example"), "Key_info": ("guidelines", "Key_info")}),
        }
        self.assertEqual(list(VARIANTS), list(expected))
        for name, (components, bindings) in expected.items():
            variant = VARIANTS[name]
            self.assertEqual((variant.components, variant.bindings, variant.prompt_key),
                             (components, bindings, name))
            self.assertTrue(variant.description.strip())
            for component, output in bindings.values():
                self.assertIn(output, COMPONENTS[component].outputs)
        self.assertEqual(COMPONENTS["patient_retrieval"].credentials, ("OPENAI_API_KEY",))
        self.assertEqual(COMPONENTS["guidelines"].credentials, ("OPENAI_API_KEY", "GRAPHRAG_API_KEY"))
        self.assertEqual(COMPONENTS["patient_retrieval"].parameters["alpha"].config_field,
                         "patient_retrieval_alpha")

    def test_imports_are_in_header_and_registry_import_loads_no_runtime_or_network(self):
        path = ROOT / "scripts/med_copilot_variants.py"
        tree = ast.parse(path.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        first_definition = min(node.lineno for node in tree.body
                               if isinstance(node, (ast.ClassDef, ast.FunctionDef)))
        self.assertTrue(all(node in tree.body and node.lineno < first_definition for node in imports))
        program = (
            "import builtins, socket, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "def blocked(*args, **kwargs): raise AssertionError('Network access attempted')\n"
            "socket.socket.connect = blocked\n"
            "socket.getaddrinfo = blocked\n"
            "original = builtins.__import__\n"
            "def guarded(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in ('torch', 'pandas', 'openai', 'litellm', 'transformers', 'graphrag', 'faiss', 'sentence_transformers', 'yaml'):\n"
            "        raise AssertionError('Runtime imported: ' + name)\n"
            "    return original(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded\n"
            "from scripts.med_copilot_variants import VARIANTS\n"
            "assert len(VARIANTS) == 4\n"
        )
        result = subprocess.run([sys.executable, "-I", "-c", program], cwd=ROOT.parent,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
