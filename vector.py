from core.help_functions import preparation, fill_info


add_documents, vector_store = preparation()
retriever = fill_info(add_documents, vector_store, "pdf_folder")
