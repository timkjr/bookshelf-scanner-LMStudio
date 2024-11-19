import { ChangeDetectionStrategy, Component, OnDestroy, signal } from "@angular/core";
import { Subscription } from "rxjs";
import { BookPredictionService } from "@/core";

@Component({
  selector: "app-root",
  standalone: true,
  templateUrl:"./app.component.html",
  styleUrl: "./app.component.scss",
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppComponent implements OnDestroy {
  private predictionSubscription: Subscription | null = null;
  public selectedFile = signal<File | null>(null);
  public results = signal<string[]>([]);
  public errorMessage = signal("");
  public isProcessing = signal(false);

  constructor(private bookPredictionService: BookPredictionService) { }

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0] ?? null;
    this.results.set([]);
    this.errorMessage.set("");
    this.isProcessing.set(false);
  }

  onSubmit() {
    if (!this.selectedFile()) {
      return;
    }

    this.isProcessing.set(true);

    this.predictionSubscription = this.bookPredictionService.startPrediction(this.selectedFile()!).subscribe({
      next: (result) => {
        this.results().push(result);
      },
      error: (error) => {
        this.errorMessage.set(error);
        this.isProcessing.set(false);
      },
      complete: () => {
        this.isProcessing.set(false);
      }
    });
  }

  cancelProcess() {
    this.bookPredictionService.cancelPrediction();

    if (this.predictionSubscription) {
      this.predictionSubscription.unsubscribe();
      this.predictionSubscription = null;
    }

    this.isProcessing.set(false);
  }

  ngOnDestroy() {
    if (this.isProcessing()) {
      this.cancelProcess();
    }
  }
}
