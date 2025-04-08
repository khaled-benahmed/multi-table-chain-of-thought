from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from dataclasses import dataclass
from jinja2 import Template

template_folder = Path(__file__).resolve().parent / "prompts"
env = Environment(loader=FileSystemLoader(template_folder))


@dataclass
class PromptTemplate:
    template: Template

    def build(self, **kwargs):
        return self.template.render(**kwargs)

    def load(name):
        template = env.get_template(f"{name}.jinja")
        return PromptTemplate(template=template)
