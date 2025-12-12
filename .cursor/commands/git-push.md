# Git Add, Commit, and Push

## Objective
Add all changes, commit them with a descriptive message, and push to the remote repository.

## Steps
1. Check the current git status to see what files have changed
2. Add all changes to staging (`git add .`)
3. Commit the changes with a descriptive message based on the changes made
4. Push the committed changes to the remote repository (`git push`)

## Instructions
- First, run `git status` to see what files have been modified
- Add all changes using `git add .`
- Create a commit message that accurately describes the changes made. If the user hasn't provided a specific commit message, infer one from the changes:
  - Review the modified files and their changes
  - Create a clear, concise commit message following conventional commit format when possible
  - If multiple unrelated changes exist, consider asking the user or creating separate commits
- Commit with the message: `git commit -m "your message here"`
- Push to the remote: `git push`

## Notes
- Always verify the changes before committing
- Use meaningful commit messages that describe what was changed and why
- If there are uncommitted changes that seem unrelated, consider asking the user if they want separate commits
