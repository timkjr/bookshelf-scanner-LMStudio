import { Injectable, OnDestroy } from "@angular/core";
import { Observable, finalize } from "rxjs";

@Injectable({
  providedIn: "root"
})
export class BookPredictionService implements OnDestroy {
  private apiUrl = "http://localhost:8000/predict";
  private abortController: AbortController | null = null;

  ngOnDestroy() {
    this.cancelPrediction();
  }

  startPrediction(file: File): Observable<string> {
    const formData = new FormData();
    formData.append("file", file);

    // Create an observable to emit streamed results
    return new Observable<string>((observer) => {
      this.abortController = new AbortController();

      fetch(this.apiUrl, {
        method: "POST",
        body: formData,
        signal: this.abortController.signal
      }).then(response => {
        if (!response.ok) {
          observer.error(`Server error: ${response.statusText}`);
          return;
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder("utf-8");

        const readStream = async () => {
          try {
            const { done, value } = await reader.read();

            if (done) {
              observer.complete();
              return;
            }

            const chunk = decoder.decode(value);
            // Split the chunk by newline to get individual results
            const lines = chunk.split("\n").filter(Boolean);
            lines.forEach(line => observer.next(line));
            await readStream();
          }
          catch (error) {
            if (this.abortController?.signal.aborted) {
              observer.error("Process canceled by user.");
            }
            else {
              observer.error(`Error reading stream: ${error}`);
            }
          }
        };

        readStream();
      }).catch(error => {
        if (this.abortController?.signal.aborted) {
          observer.error("Process canceled by user.");
        }
        else {
          observer.error(`Fetch error: ${error}`);
        }
      });
    }).pipe(
      finalize(() => {
        // Clean up when the observable completes or errors
        this.abortController = null;
      })
    );
  }

  cancelPrediction() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }
}
