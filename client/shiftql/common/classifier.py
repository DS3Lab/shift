from pydantic import BaseModel, Field


class GradientDescentSettings(BaseModel):
    learning_rate: float = 0.01
    num_epochs: int = 2000


class Classifier(BaseModel):
    name: str
    parameters: GradientDescentSettings = Field(
        GradientDescentSettings(), alias="parameters"
    )

    @property
    def value(self):
        if self.name in ["Euclidean NN", "Cosine NN"]:
            return self.name
        else:
            if self.parameters is None:
                self.parameters = GradientDescentSettings()
            return "{}({})".format(self.name, self.parameters)

    @staticmethod
    def parse(name, parameters):
        if parameters is None:
            parameters = GradientDescentSettings()
        return Classifier(name=name, parameters=parameters)
