"""Tests for index reconciliation.

`reconcile_indexes` runs at startup before Beanie initialises, so anything it
raises fails the boot. These cover reading a model's declarations without a
database.
"""

from beanie import Document
from pymongo import IndexModel

from app.core.db.index_migrations import (
    collection_name,
    declared_settings,
    declared_ttl_indexes,
)


class ModelWithoutSettings(Document):
    field: str = ""


class ModelWithSettings(Document):
    field: str = ""

    class Settings:
        name = "explicit_collection"
        indexes = [
            IndexModel([("field", 1)], name="plain_index"),
            IndexModel([("field", 1)], name="ttl_index", expireAfterSeconds=99),
        ]


class ModelWithAwkwardIndexes(Document):
    field: str = ""

    class Settings:
        indexes = [
            "a_plain_string_index",
            IndexModel([("field", 1)], name="float_ttl", expireAfterSeconds=99.0),
            IndexModel([("field", 1)], name="junk_ttl", expireAfterSeconds="soon"),
            IndexModel([("field", 1)], name="", expireAfterSeconds=5),
        ]


class TestDeclaredSettings:
    def test_a_model_without_settings_reads_as_none(self) -> None:
        assert declared_settings(ModelWithoutSettings) is None

    def test_a_model_with_settings_reads_the_inner_class(self) -> None:
        assert declared_settings(ModelWithSettings) is ModelWithSettings.Settings


class TestCollectionName:
    def test_the_declared_name_wins(self) -> None:
        assert collection_name(ModelWithSettings) == "explicit_collection"

    def test_it_falls_back_to_the_class_name(self) -> None:
        assert collection_name(ModelWithoutSettings) == "modelwithoutsettings"


class TestDeclaredTtlIndexes:
    def test_only_ttl_indexes_are_reported(self) -> None:
        assert declared_ttl_indexes(ModelWithSettings) == {"ttl_index": 99}

    def test_a_model_without_settings_declares_nothing(self) -> None:
        assert declared_ttl_indexes(ModelWithoutSettings) == {}

    def test_a_float_ttl_is_still_reconciled(self) -> None:
        assert declared_ttl_indexes(ModelWithAwkwardIndexes)["float_ttl"] == 99

    def test_a_non_numeric_ttl_is_skipped_rather_than_crashing(self) -> None:
        assert "junk_ttl" not in declared_ttl_indexes(ModelWithAwkwardIndexes)

    def test_an_unnamed_index_is_skipped(self) -> None:
        assert "" not in declared_ttl_indexes(ModelWithAwkwardIndexes)

    def test_non_indexmodel_entries_are_ignored(self) -> None:
        assert set(declared_ttl_indexes(ModelWithAwkwardIndexes)) == {"float_ttl"}
