A conversational LLM can be used as a classifier by prompting it to respond to a sentence using 'yes' or 'no', or any other set of categories.

Probably we don't actually have to emit any output either. Prompting the LLM appropriately should put it in a state which can be used like an embedding, where we can classify this embedding using distance from a precalculated position for 'yes' or 'no'.
