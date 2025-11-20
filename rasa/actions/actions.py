# actions/actions.py
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionTranslateText(Action):
    
    def name(self) -> Text:
        return "action_translate_text"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Obtener el último mensaje del usuario
        user_message = tracker.latest_message.get('text')
        
        # Diccionario de traducción simple (puedes expandirlo)
        translations = {
            'perro': 'uchete',
            'gato': 'Sufejuimís Michi',
            'tigre': 'Ino',
            'casa': 'Sobo',
            'lagarto': 'Secque, Cappue',
            'pájaro': 'Isa',
            'bien': 'Accu',
            'día':'Nete',
            'dia':'Nete',
            'sembrar':'Banaqui'
        }
        
        # Buscar traducción
        message = user_message.lower()
        translated = None
        
        for spanish, shipibo in translations.items():
            if spanish in message:
                translated = f"'{spanish}' en shipibo-konibo es '{shipibo}'"
                break
        
        if not translated:
            translated = f"No tengo la traducción para '{user_message}' aún. ¿Quieres que aprenda más palabras?"
        
        dispatcher.utter_message(text=translated)
        return []


class ActionCheckGrammar(Action):
    
    def name(self) -> Text:
        return "action_explain_grammar"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text')
        
        # Aquí podrías integrar un servicio de corrección gramatical
        response = f"Revisando gramática de: '{user_message}'"
        respuesta = (
                    "En shipibo-konibo, la estructura básica de la oración es Sujeto + Objeto + Verbo.\n"
                    "Por ejemplo: 'Noya ani' significa 'El hombre come'.\n"
                    "Los pronombres personales son: ja (yo), mi (tú), i (él/ella)."
                                                            )
        dispatcher.utter_message(text=respuesta)
        return []
