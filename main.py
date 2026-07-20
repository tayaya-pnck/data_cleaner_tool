from core import (
    DataRaw,
    BaseCleaner,
    ColumnDropper,
    Deduplicator,
    DataTypeConverter,
    AllowedValuesFilter,
    StringNormalizer,
    MissingValueHandler,
    OutlierHandler
)
from config import CleaningConfig, ColumnConfig
from typing import Optional, Tuple
import pandas as pd
from report import CleaningReport
from strategy import MissingStrategy, OutlierStrategy

class DataCleaner:
    DEFAULT_FLOWS = [
        ColumnDropper,
        Deduplicator,
        DataTypeConverter,
        AllowedValuesFilter,
        StringNormalizer,
        MissingValueHandler,
        OutlierHandler
    ]

    def __init__(self, config: CleaningConfig, data_raw: DataRaw, flows: Optional[list[type[BaseCleaner]]] = None):
        self.config = config
        self.flows = [flow(config) for flow in (flows or self.DEFAULT_FLOWS)]
        dict_col = {}
        for conf in self.config.column_configs:
            try:
                dict_col[conf.name] = getattr(data_raw, conf.name)
            except Exception as e:
                raise ValueError(f"Unable to extract attribute named {conf.name}: {e}")

        self.df = pd.DataFrame(dict_col)

    def clean(self) -> Tuple[pd.DataFrame, CleaningReport]:
        df = self.df
        report = CleaningReport(
             initial_rows=len(df),
             initial_cols=len(df.columns)
        )

        for flow in self.flows:
            df = flow.implement(df, report)

        report.final_cols = len(df.columns)
        report.final_rows = len(df)
        return df.reset_index(drop=True), report

if __name__ == "__main__":
    data_raw = DataRaw(
        n_data=5,
        name=["Baskara", "Jakaria", "Jahari", pd.NA, "Indah"],
        age=[28, 24, 26, 22, 26],
        gender=["M", "M    ", "X", "M", pd.NA]
    )

    config = CleaningConfig(
        drop_col_threshold=0.3,
        column_configs=[
            ColumnConfig(
                name="name",
                dtype="str",
                missing_strategy=MissingStrategy.CONSTANT,
                fill_value="Unknown",
                strip_string=True
            ),
            ColumnConfig(
                name="age",
                dtype="int",
                missing_strategy=MissingStrategy.MEAN,
                outlier_strategy=OutlierStrategy.IQR
            ),
            ColumnConfig(
                name="gender",
                dtype="str",
                missing_strategy=MissingStrategy.MODE,
                strip_string=True,
                lower_case=True,
                allowed_values=["M", "F"]
            )
        ],
    )

    cleaner = DataCleaner(config, data_raw)
    clean_df, report = cleaner.clean()

    print(clean_df)
    print(report.generate())