import {Injectable, OnDestroy} from "@angular/core";
import {Observable, finalize} from "rxjs";
import {Result} from "@/core/models";

@Injectable({
  providedIn: "root",
})
export class BookPredictionService implements OnDestroy {
  private apiUrl = "http://localhost:8000/predict";
  private abortController: AbortController | null = null;

  ngOnDestroy() {
    this.cancelPrediction();
  }

  startPrediction(file: File): Observable<Result<string>> {
    const formData = new FormData();
    formData.append("file", file);

    // Create an observable to emit streamed results
    return new Observable<Result<string>>((observer) => {
      this.abortController = new AbortController();

      fetch(this.apiUrl, {
        method: "POST",
        body: formData,
        signal: this.abortController.signal,
      })
        .then((response) => {
          if (!response.ok) {
            observer.error(`Server error: ${response.statusText}`);
            return;
          }

          const reader = response.body!.getReader();
          const decoder = new TextDecoder("utf-8");
          let buffer = "";

          const readStream = async () => {
            try {
              const {done, value} = await reader.read();

              if (done) {
                observer.complete();
                return;
              }

              const chunk = decoder.decode(value);
              // Split the chunk by newline to get individual results
              buffer += chunk;

              const lines = buffer.split("\n");
              buffer = lines.pop()!; // Keep the last partial line

              for (const line of lines) {
                if (line.trim() === "") {
                  continue;
                }

                try {
                  const data: Result<string> = JSON.parse(line);
                  observer.next(data);
                } catch (e) {
                  observer.error(`Error parsing JSON: ${e}`);
                }
              }

              await readStream();
            } catch (error) {
              if (this.abortController?.signal.aborted) {
                observer.error("Process canceled by user.");
              } else {
                observer.error(`Error reading stream: ${error}`);
              }
            }
          };

          readStream();
        })
        .catch((error) => {
          if (this.abortController?.signal.aborted) {
            observer.error("Process canceled by user.");
          } else {
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
