from typing import Callable


def get_encoders_decoders(
    text: str,
) -> tuple[Callable, Callable, int]:
    characters_in_text = sorted(list(set(text)))
    map_characters_to_integers = {
        characters: integers for integers, characters in enumerate(characters_in_text)
    }
    map_integers_to_characters = {
        integers: characters for integers, characters in enumerate(characters_in_text)
    }

    return (
        lambda string: [
            map_characters_to_integers[characters] for characters in string
        ],
        lambda list_integers: "".join(
            [map_integers_to_characters[integer] for integer in list_integers]
        ),
        len(characters_in_text),
    )
