from pydantic import BaseModel, Field, model_validator
from abc import ABC, abstractmethod
from config import CleaningConfig
import pandas as pd
from report import CleaningReport
from strategy import OutlierStrategy, MissingStrategy
import numpy as np

class DataRaw(BaseModel):
    n_data: int
    name: list = Field(default_factory=list)
    age: list = Field(default_factory=list)
    gender: list = Field(default_factory=list)

    @model_validator(mode='after')
    def check_length(self) -> "DataRaw":
        if (len(self.name) + len(self.age) + len(self.gender)) != (3*self.n_data):
            raise ValueError(f"Panjang list semua data harus sama dengan n_data yaitu {self.n_data}")
        return self

class BaseCleaner(ABC):
    def __init__(self, config: CleaningConfig) -> None:
        self.config = config
    
    @abstractmethod
    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        ...

class OutlierHandler(BaseCleaner):
    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        for config in self.config.column_configs:
            if config.name not in df.columns or config.outlier_strategy == OutlierStrategy.NONE:
                continue
            
            series = df[config.name]
            if not pd.api.types.is_numeric_dtype(series):
                continue

            if config.outlier_strategy == OutlierStrategy.IQR:
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            else: # zscore
                mean, std = series.mean(), series.std()
                if std == 0 or pd.isna(std):
                    continue
                lower, upper = mean-3*std, mean+3*std

            mask = (series < lower) | (series > upper)
            n = int(mask.sum())
            if n:
                df.loc[mask, config.name] = np.nan
                df[config.name] = df[config.name].fillna(series.median())
                report.outlier_handled[config.name] = n

        return df

class ColumnDropper(BaseCleaner):
    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        dropped =[
            col for col in df.columns if df[col].isna().mean() > self.config.drop_col_threshold
        ]

        report.columns_dropped = dropped
        return df.drop(columns=dropped)

class DataTypeConverter(BaseCleaner):
    # this will translate dtype from our config
    _DTYPE_MAP = {
        "int": "int64",
        "float": "float64",
        "str": "string",
        "bool": "boolean",
        "datetime": "datetime64[ns]"
    }

    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        for config in self.config.column_configs:
            if not config.dtype or config.name not in df.columns:
                continue
            na_before = df[config.name].isna().sum()

            try:
                if config.dtype == "datetime":
                    df[config.name] = pd.to_datetime(df[config.name], errors="coerce")
                else:
                    df[config.name] = pd.to_numeric(df[config.name], errors="coerce") if config.dtype in ("int","float") else df[config.name]
                    df[config.name] = df[config.name].astype(self._DTYPE_MAP[config.dtype])
                
                na_after = df[config.name].isna().sum()
                conversion_error = int(na_after - na_before)
                if conversion_error > 0:
                    report.type_conversion_failed[config.name] = conversion_error
            except Exception as e:
                report.type_conversion_failed_reason[config.name] = e
                report.type_conversion_failed[config.name] = -1

        return df

class MissingValueHandler(BaseCleaner):
    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        for config in self.config.column_configs:
            if config.name not in df.columns or config.missing_strategy == MissingStrategy.NONE:
                continue
            n_missing = int(df[config.name].isna().sum())
            if n_missing == 0:
                continue

            strategy = config.missing_strategy
            if strategy == MissingStrategy.DROP:
                df = df.dropna(subset=[config.name])
            elif strategy == MissingStrategy.MEAN:
                df[config.name] = df[config.name].fillna(df[config.name].mean())
            elif strategy == MissingStrategy.MEDIAN:
                df[config.name] = df[config.name].fillna(df[config.name].median())
            elif strategy == MissingStrategy.MODE:
                mode = df[config.name].mode()
                if not mode.empty:
                    df[config.name] = df[config.name].fillna(mode.iloc[0])
            elif strategy == MissingStrategy.BFILL:
                df[config.name] = df[config.name].bfill()
            elif strategy == MissingStrategy.FFILL:
                df[config.name] = df[config.name].ffill()
            elif strategy == MissingStrategy.BFILL_TO_FFILL:
                df[config.name] = df[config.name].bfill()
                df[config.name] = df[config.name].ffill()
            elif strategy == MissingStrategy.FFILL_TO_BFILL:
                df[config.name] = df[config.name].ffill()
                df[config.name] = df[config.name].bfill()
            elif strategy == MissingStrategy.CONSTANT:
                df[config.name] = df[config.name].fillna(config.fill_value)
            
            report.missing_handled[config.name] = n_missing

        return df

class Deduplicator(BaseCleaner):
    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        
        # PROJECT: BUATLAH KODE UNTUK MENGHAPUS DUPLIKASI DATA DISINI

        # NOTES:
        # 1. Pertimbangkan atribbut drop_duplicated pada ColumnConfig
        # 2. Pertimbangkan kemungkinan user menghapus data berdasarkan kolom subset tertentu.
        if self.config.drop_duplicated:
            before = len(df)
            df = df.drop_duplicates(subset=self.config.duplicate_subset)
            report.duplicated_removed = before - len(df)

        return df

class StringNormalizer(BaseCleaner):
    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        # PROJECT: BUAT KODE UNTUK MELAKUKAN STRING NORMALISASI:
        # Menghapus extra spasi
        # Mengubah huruf menjadi huruf kecil (lower case)

        # HINTS:
        # Perhatikan apakah sedang memproses kolom yang benar
        # perhatikan apakah type data adalah string (object di pandas)
        # perhatikan apakah config mengijinkan text stripping
        # perhatikan apakah config ingin melakukan perubahan ke lower case.
        for c in self.config.column_configs:
            if c.name not in df.columns:
                continue
            if pd.api.types.is_string_dtype(df[c.name]):
                if c.strip_string:
                    df[c.name] = df[c.name].str.strip()
                if c.lower_case:
                    df[c.name] = df[c.name].str.lower()
        return  df

class AllowedValuesFilter(BaseCleaner):
    def implement(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        # PORJECT: BUATLAH KODE UNTUK MEMFILTER VALUES YANG HANYA DIIJINKAN (menjadikannya np.nan)
        # CONTOH, KOLOM GENDER, KITA HANYA MENGIJINKAN Male dan Female, maka
        # values selain itu akan dibuang (dijadikan np.nan)

        # HINTS:
        # perhatikan apakah sedang memproses kolom yang benar
        # perhatikan apakah user memiliki allowed_values spesifik atau kolom bisa menerima value apa saja.
        # ambillah value yang diluar allowed_values untuk diubah menjadi np.nan.
        
        return df