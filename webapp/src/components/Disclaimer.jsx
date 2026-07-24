import { useEffect, useState } from "react";

const ACK_KEY = "pneumo_disclaimer_ack_v1";

// The legal disclaimer. Two parts:
//  1) A blocking modal shown on first visit that must be acknowledged.
//  2) A persistent banner at the top of the app (always visible thereafter).
export function DisclaimerGate({ children }) {
  const [ack, setAck] = useState(true); // assume acked until we read storage

  useEffect(() => {
    try {
      setAck(localStorage.getItem(ACK_KEY) === "1");
    } catch {
      setAck(false);
    }
  }, []);

  function accept() {
    try {
      localStorage.setItem(ACK_KEY, "1");
    } catch {
      /* ignore */
    }
    setAck(true);
  }

  return (
    <>
      {!ack && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="disc-title">
          <div className="modal">
            <h2 id="disc-title">⚠️ Educational demo — not a medical device</h2>
            <p>
              This is a <strong>personal portfolio / practice project</strong>. The model
              behind it is <strong>largely untested</strong> and has <strong>not</strong> been
              validated, approved, or certified for any clinical use.
            </p>
            <ul>
              <li>Do <strong>not</strong> use it to make or support any medical decision.</li>
              <li>Results can be wrong in both directions (missed and false findings).</li>
              <li>Do not upload real patient data or anything that identifies a person.</li>
            </ul>
            <p>
              For any health concern, consult a qualified physician. By continuing you confirm
              you understand this is a demonstration only.
            </p>
            <button className="btn btn-primary" onClick={accept}>
              I understand — continue
            </button>
          </div>
        </div>
      )}
      {children}
    </>
  );
}

export function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner" role="note">
      <strong>Demo only —</strong> largely untested research project. Not a medical device;
      do not use for medical decisions and do not upload identifiable patient data.
    </div>
  );
}
