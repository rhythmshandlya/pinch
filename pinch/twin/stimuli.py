"""What you do to Pinch in the twin: talk, touch, make sounds."""



class TwinStimuli:
    def say(self, text):
        self.user_active = True
        self._rec("say", text=text)
        if "audio" in self.faults:                       # (the microphones are dead: nothing heard)
            return None
        return self.mind.hear_text(text)

    def touch(self, gesture):
        self._rec("touch", gesture=gesture)
        self.mind.touch(gesture)

    def hear_audio(self, chunk):
        if "audio" in self.faults:
            return
        for s in self.hearing.feed(chunk):
            self.mind.hear_summary(s)
        for kind, _ in self.hearing.take_events():
            self.mind.hear_event(kind)
