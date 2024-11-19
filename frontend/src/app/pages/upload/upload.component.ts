import {Component, OnDestroy, signal} from "@angular/core";
import {FormControl, FormGroup, ReactiveFormsModule} from "@angular/forms";
import {Subscription} from "rxjs";
import {BookPredictionService} from "@/core";

@Component({
  selector: "app-upload",
  standalone: true,
  templateUrl: "./upload.component.html",
  styleUrl: "./upload.component.scss",
  imports: [ReactiveFormsModule],
})
export class UploadComponent implements OnDestroy {
  private predictionSubscription: Subscription | null = null;
  public readonly uploadForm: FormGroup<UploadForm>;
  public selectedFile = signal<File | null>(null);
  public results = signal<string[]>([]);
  public errorMessage = signal("");
  public isProcessing = signal(false);

  constructor(private bookPredictionService: BookPredictionService) {
    this.uploadForm = new FormGroup<UploadForm>({
      image: new FormControl<File | null>(null),
    });
  }

  ngOnDestroy() {
    if (this.isProcessing()) {
      this.cancelProcess();
    }
  }

  onFileSelected(event: Event) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.selectedFile = (event.target as any).files[0] ?? null;
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
      },
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
}

interface UploadForm {
  image: FormControl<File | null>;
}
