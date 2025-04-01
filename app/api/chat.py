# app/api/chat.py
from flask_restful import Resource
from flask import request, jsonify, session
from app.services.chatbot import get_response
# from app.services.celery_tasks import train_model_async # Not needed if triggering training manually

class ChatAPI(Resource):
    def post(self):
        try:
            data = request.get_json()
            user_message = data.get('message', '') # Use .get for safety
            if not user_message:
                 return jsonify({'error': 'No message provided'}), 400

            # Initialize conversation history if not present
            if 'conversation_history' not in session:
                session['conversation_history'] = []

            # Limit conversation history length (optional)
            MAX_HISTORY = 10 # Keep last 10 turns (user + bot)
            session['conversation_history'] = session['conversation_history'][-MAX_HISTORY:]

            session['conversation_history'].append({'role': 'user', 'content': user_message})
            session.modified = True

            # Get both the main response and the conversational explanation
            response_text, conv_explanation = get_response(session['conversation_history'])

            # Append main response to history
            full_bot_message = response_text
            session['conversation_history'].append({'role': 'bot', 'content': full_bot_message})
            session.modified = True

            # Return both parts separately to the frontend
            return jsonify({'response': response_text, 'explanation_detail': conv_explanation or ""}) # Ensure explanation is string

        except Exception as e:
            print(f"Error in ChatAPI: {e}")
            traceback.print_exc() # Print full traceback for server errors
            return jsonify({'error': 'An internal server error occurred'}), 500