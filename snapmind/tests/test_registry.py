import pytest

from snapmind.core.registry import Registry, RegistryError


class TestRegistryBasics:
    def test_register_and_create(self):
        reg = Registry("test", expected_type=object)

        @reg.register("my_component")
        class MyComponent:
            def __init__(self):
                self.value = 42

        instance = reg.create("my_component")
        assert isinstance(instance, MyComponent)
        assert instance.value == 42

    def test_create_unknown_key_raises(self):
        reg = Registry("test", expected_type=object)
        with pytest.raises(RegistryError, match="unknown.*test"):
            reg.create("does_not_exist")

    def test_duplicate_register_raises(self):
        reg = Registry("test", expected_type=object)

        @reg.register("dup")
        class A:
            pass

        with pytest.raises(RegistryError, match="already registered.*dup"):

            @reg.register("dup")
            class B:
                pass

    def test_duplicate_register_with_override(self):
        reg = Registry("test", expected_type=object)

        @reg.register("dup")
        class A:
            pass

        @reg.register("dup", override=True)
        class B:
            pass

        instance = reg.create("dup")
        assert isinstance(instance, B)

    def test_list_registered_keys(self):
        reg = Registry("test", expected_type=object)

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        keys = reg.list()
        assert "a" in keys
        assert "b" in keys

    def test_list_is_copy(self):
        reg = Registry("test", expected_type=object)

        @reg.register("a")
        class A:
            pass

        keys = reg.list()
        keys.append("b")
        assert "b" not in reg.list()

    def test_contains(self):
        reg = Registry("test", expected_type=object)

        @reg.register("present")
        class A:
            pass

        assert "present" in reg
        assert "absent" not in reg


class TestRegistryKwargs:
    def test_create_passes_kwargs(self):
        reg = Registry("test", expected_type=object)

        @reg.register("with_args")
        class WithArgs:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        instance = reg.create("with_args", x=10, y=20)
        assert instance.x == 10
        assert instance.y == 20

    def test_create_missing_kwargs_raises(self):
        reg = Registry("test", expected_type=object)

        @reg.register("needs_arg")
        class NeedsArg:
            def __init__(self, required):
                pass

        with pytest.raises(TypeError):
            reg.create("needs_arg")


class TestRegistryTypeSafety:
    def test_register_wrong_type_raises(self):
        class Base:
            pass

        class Impl:
            pass

        reg = Registry("test", expected_type=Base)
        with pytest.raises(TypeError, match="must subclass"):
            reg.register("wrong", Impl)

    def test_register_correct_type_succeeds(self):
        class Base:
            pass

        class Impl(Base):
            pass

        reg = Registry("test", expected_type=Base)
        reg.register("correct", Impl)
        instance = reg.create("correct")
        assert isinstance(instance, Impl)
